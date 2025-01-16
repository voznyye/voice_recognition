import os
import json
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Функция извлечения признаков
def extract_features(audio_path, n_mels=80):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)(waveform)
    mel_spec = mel_spec.squeeze(0).transpose(0, 1)
    return mel_spec

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

# Пользовательский Dataset для LibriSpeech
class CustomDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path = sample['audio_path']
        features = extract_features(audio_path)
        label = sample['label']
        return features, label

# Функция преобразования транскрипта в тензор
def transcript_to_tensor(transcript, label_map):
    first_char = transcript[0]
    if first_char not in label_map:
        label_map[first_char] = len(label_map)
    return label_map[first_char]

# Функция для обработки батчей
def collate_fn(batch, input_dim):
    label_map = {}
    inputs = []
    labels = []
    max_length = max([sample[0].size(0) for sample in batch])

    for sample in batch:
        features, label = sample
        padded_features = F.pad(features, (0, 0, 0, max_length - features.size(0)))
        inputs.append(padded_features)
        label_tensor = torch.tensor(transcript_to_tensor(label, label_map), dtype=torch.long)
        labels.append(label_tensor)

    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels

# Функция подсчета метрик
def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)
    return precision, recall, f1, cm

# Функция для тестирования модели
def test_lstm_model(test_dataset_path, model_path, input_dim=80, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = CustomDataset(test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, input_dim))

    model = LSTMModel(input_dim=input_dim, hidden_dim=64, output_dim=28).to(device)  # Измените output_dim на 28
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    precision, recall, f1, cm = calculate_metrics(all_true_labels, all_predicted_labels)

    # Вывод метрик
    print("\n=== Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Подготовка тестирования
test_dataset_path = "./datasets/test_clean.json"
model_path = "./trained_lstm_model/lstm_model.pth"
test_lstm_model(test_dataset_path, model_path)
