import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.nn import functional as F

# Dataset class for LibriSpeech
def collate_fn(batch):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    waveforms = [item[0].numpy() for item in batch]
    texts = [item[1] for item in batch]

    # Process the audio waveform
    inputs = processor(waveforms, return_tensors="pt", padding=True, sampling_rate=16000).input_values

    # Tokenize the text labels separately
    tokenizer = processor.tokenizer
    max_length = 256  # You can adjust this value based on your needs
    labels = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids

    return inputs, labels


class LibriSpeechDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".trans.txt"):
                    with open(os.path.join(root, file), "r") as f:
                        for line in f:
                            audio_id, text = line.strip().split(" ", 1)
                            audio_path = os.path.join(root, f"{audio_id}.flac")
                            if os.path.exists(audio_path):
                                self.data.append((audio_path, text))
        if not self.data:
            raise ValueError(f"No valid data found in {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, text = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform.squeeze(0), text

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, predicted_labels)
    return precision, recall, f1, cm

# Testing function
def test_transformer_model(test_dataset_path, model_path, processor_path, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = LibriSpeechDataset(test_dataset_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to(device)
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the predicted ids
            predicted_text = processor.decode(predicted_ids[0])

            # Append the true and predicted texts
            true_text = processor.tokenizer.decode(labels[0], skip_special_tokens=True)
            all_true_labels.append(true_text)
            all_predicted_labels.append(predicted_text)

    # Calculate metrics
    precision, recall, f1, cm = calculate_metrics(all_true_labels, all_predicted_labels)

    # Print the results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    test_dataset_path = "./test_data/LibriSpeech/test-clean"
    model_path = "./trained_transformer_model"
    processor_path = "./trained_transformer_model"

    test_transformer_model(test_dataset_path, model_path, processor_path)
