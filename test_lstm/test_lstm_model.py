import torch
import torchaudio
import sacrebleu
from jiwer import wer, cer
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

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

# Creating label map
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

# Function to convert indices to text
def indices_to_text(indices, label_map):
    return [list(label_map.keys())[list(label_map.values()).index(idx)] for idx in indices]

# Model evaluation using BLEU, WER, and CER
def evaluate_model(model, dataloader, label_map, device='cuda'):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, dim=1)
            
            # Converting indices to text
            pred_text = indices_to_text(predicted_labels.tolist(), label_map)
            true_text = indices_to_text(labels.tolist(), label_map)
            
            predictions.extend(pred_text)
            references.extend(true_text)
    
    # BLEU
    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    print(f"BLEU score: {bleu_score.score:.2f}")

    # WER and CER
    wer_score = wer(references, predictions)
    cer_score = cer(references, predictions)
    
    print(f"WER: {wer_score * 100:.2f}%")
    print(f"CER: {cer_score * 100:.2f}%")

# Example usage
if __name__ == "__main__":
    dataset_path = "./datasets/train_clean_100.json"
    label_map = create_label_map(dataset_path)
    output_dim = len(label_map)

    dataset = CustomDataset(dataset_path, label_map, device='cuda')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=lambda batch: collate_fn(batch, input_dim=80, device='cuda'))

    # Loading the model
    model = LSTMModel(input_dim=80, hidden_dim=128, output_dim=output_dim).to('cuda')
    model.load_state_dict(torch.load("./trained_lstm_model/lstm_model.pth"))

    # Model evaluation
    evaluate_model(model, dataloader, label_map, device='cuda')
