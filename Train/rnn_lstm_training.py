import os
import json
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
import time

# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Function to extract features on GPU
def extract_features(audio_path, n_mels=80, device='cpu'):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)(waveform).to(device)  # Moving to GPU
    mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # (time, freq) shape
    return mel_spec

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Last timestep
        out = self.fc(lstm_out)
        return out

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data_path, label_map, device):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.label_map = label_map
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path = sample['audio_path']

        features = extract_features(audio_path, device=self.device)  # Moving feature extraction to GPU
        label = self.label_map[sample['label']]
        return features, label

# Creating the label map
def create_label_map(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    label_map = {}
    for sample in data:
        label = sample['label']
        if label not in label_map:
            label_map[label] = len(label_map)
    return label_map

# Batch processing
def collate_fn(batch, input_dim, device):
    inputs = []
    labels = []
    max_length = max([sample[0].size(0) for sample in batch])

    for features, label in batch:
        padded_features = F.pad(features, (0, 0, 0, max_length - features.size(0)))
        inputs.append(padded_features)
        labels.append(label)

    inputs = torch.stack(inputs).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return inputs, labels

# Model training
def train_lstm_model(dataset_path, save_path, input_dim=80, hidden_dim=128, num_epochs=100, batch_size=128, num_workers=0):
    device = torch.device('cuda')

    # Creating label map
    label_map = create_label_map(dataset_path)
    output_dim = len(label_map)

    # Loading the dataset
    dataset = CustomDataset(dataset_path, label_map, device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Setting num_workers to 0 for Windows
        collate_fn=lambda batch: collate_fn(batch, input_dim, device),  # Replaced with a regular function
    )

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        start_time = time.time()  # Remembering the start time of the epoch

        model.train()
        total_loss = 0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)

            # Checking the label range
            if (labels < 0).any() or (labels >= output_dim).any():
                raise ValueError(f"Labels out of range: {labels}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Logging
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        end_time = time.time()  # Remembering the end time of the epoch
        epoch_duration = end_time - start_time  # Time taken for the epoch

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_duration:.2f}s")

    # Saving the model and label map
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "lstm_model.pth"))
    with open(os.path.join(save_path, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    print("Model and label map saved!")

# Paths
dataset_path = "./datasets/train_clean_100.json"
save_directory = "./trained_lstm_model1"

# Training
if __name__ == "__main__":
    train_lstm_model(dataset_path, save_path=save_directory)
