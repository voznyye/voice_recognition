import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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

def train_transformer_model(dataset_path, save_path):
    dataset = LibriSpeechDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CTCLoss()

    # Gradient Accumulation
    accumulate_steps = 4  # Accumulate gradients over 4 steps
    scaler = torch.amp.GradScaler()  # For mixed precision

    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        model.train()
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training with torch.amp.autocast(enabled=True)
            with torch.amp.autocast(device_type='cuda', enabled=True):  # Mixed precision
                outputs = model(inputs).logits
                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long).to(device)
                target_lengths = torch.full((labels.size(0),), labels.size(1), dtype=torch.long).to(device)

            # Compute loss without autocast (ctc_loss doesn't support Half precision)
            loss = criterion(outputs.permute(1, 0, 2).float(), labels, input_lengths, target_lengths)

            scaler.scale(loss).backward()  # Backpropagate with mixed precision
            if (i + 1) % accumulate_steps == 0:
                scaler.step(optimizer)  # Update the model
                scaler.update()  # Update the scaler

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")

    # Save the model and processor after training
    print("Saving model and processor...")
    model.save_pretrained(save_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    processor.save_pretrained(save_path)

if __name__ == "__main__":
    save_directory = "./trained_transformer_model"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    train_transformer_model(f"dev-clean/LibriSpeech/dev-clean", save_directory)
