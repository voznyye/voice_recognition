import os
import json
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Функция извлечения признаков
def extract_features(audio_path, n_mels=80):  # Устанавливаем 80 мел-частотных коэффициентов
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)(waveform)  # 80 мел-частотных коэффициентов
    mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # (time, freq) shape
    return mel_spec

# Модель LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM возвращает выход и скрытые состояния
        lstm_out = lstm_out[:, -1, :]  # Используем выход последнего временного шага
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
        features = extract_features(audio_path)  # Извлекаем признаки из аудио
        label = sample['label']
        return features, label

# Функция преобразования транскрипта в тензор
def transcript_to_tensor(transcript, label_map):
    first_char = transcript[0]
    
    # Проверяем, есть ли символ в маппинге
    if first_char not in label_map:
        label_map[first_char] = len(label_map)  # Присваиваем новый индекс для символа
    
    # Возвращаем индекс символа
    return label_map[first_char]

# Функция для обработки батчей
def collate_fn(batch, input_dim, device):
    inputs = []
    labels = []
    max_length = max([sample[0].size(0) for sample in batch])

    for sample in batch:
        features, label = sample
        padded_features = F.pad(features, (0, 0, 0, max_length - features.size(0)))  # Дополняем до максимальной длины
        inputs.append(padded_features)

        label_tensor = transcript_to_tensor(label, label_map)
        labels.append(label_tensor)  # Добавляем числовой индекс, а не строку

    inputs = torch.stack(inputs).to(device)  # Перемещаем данные на GPU
    labels = torch.tensor(labels, dtype=torch.long).to(device)  # Перемещаем метки на GPU

    return inputs, labels

# Функция для подсчета количества классов
def get_output_dim(labels):
    return len(set(labels))  # количество уникальных классов

# Функция обучения модели
def train_lstm_model(dataset_path, save_path, input_dim=80, hidden_dim=64, num_epochs=100, batch_size=16):
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем датасет
    dataset = CustomDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, input_dim, device))

    # Получаем метки из датасета
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels)

    output_dim = get_output_dim(all_labels)

    # Создаем модель с правильным количеством выходных классов
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)  # Перемещаем модель на GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Тренировка модели
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)  # Перемещаем данные на GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")
    
    # Сохраняем модель
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "lstm_model.pth"))
    print("Model saved!")


# Инициализация пустой карты меток
label_map = {}

# Подготовка датасета и обучение модели
dataset_path = "./datasets/train_clean_100.json"
save_directory = "./trained_lstm_model"  # Указываем путь для сохранения модели
train_lstm_model(dataset_path, save_path=save_directory)
