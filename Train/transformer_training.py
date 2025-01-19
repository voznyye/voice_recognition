import os
import time
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import multiprocessing
import os
import pandas as pd

# ===== Step 1: Generate TSV File =====
def generate_tsv(librispeech_dir, output_tsv):
    """
    Generates a TSV file for the LibriSpeech dataset.
    Args:
        librispeech_dir (str): Path to the LibriSpeech directory (e.g., train-clean-100).
        output_tsv (str): Path to save the TSV file.
    """
    data = []

    for root, _, files in os.walk(librispeech_dir):
        # Searching for transcription files
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(root, file)

                # Reading transcriptions
                with open(transcript_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            audio_id, text = parts
                            audio_path = os.path.join(root, f"{audio_id}.flac")

                            if os.path.exists(audio_path):
                                data.append({"path": audio_path, "sentence": text})
                            else:
                                print(f"Audio file not found: {audio_path}")

    # Create DataFrame and save it as TSV
    df = pd.DataFrame(data)
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"TSV file successfully saved: {output_tsv}")

librispeech_dir = "./test-other/LibriSpeech/test-other"  # Path to the LibriSpeech directory

original_tsv = "./test-other.tsv"  # Generated TSV file


# ===== Step 2: Dataset =====
class CommonVoiceDataset(Dataset):
    def __init__(self, tsv_path, audio_dir, preprocess=False):
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        self.data = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir
        self.preprocess = preprocess

        required_columns = {"path", "sentence"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"TSV file must contain columns: {required_columns}")

        self.data = self.data.dropna(subset=["path", "sentence"])
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {tsv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row["path"]
        text = row["sentence"]

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        return waveform.squeeze(0), text


# ===== Step 3: Data Preparation Function =====
def collate_fn(batch):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    waveforms = [item[0].numpy() for item in batch]
    texts = [item[1] for item in batch]

    inputs = processor(
        waveforms, return_tensors="pt", padding=True, sampling_rate=16000
    ).input_values

    labels = processor.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).input_ids

    return inputs, labels


# ===== Step 4: Model Training =====
def train_transformer_model(tsv_path, audio_dir, save_path, config):
    """
    Main function to train the model.
    """
    torch.backends.cudnn.benchmark = True

    dataset = CommonVoiceDataset(tsv_path, audio_dir, preprocess=False)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(8, multiprocessing.cpu_count() - 1),  # Increasing worker threads
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    criterion = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id)
    scaler = torch.amp.GradScaler()  # Fixed

    log_file = os.path.join(save_path, "training_log.txt")
    total_start_time = time.time()  # Start of total time

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        model.train()
        epoch_start_time = time.time()  # Epoch start time
        epoch_loss = 0

        for i, (inputs, labels) in enumerate(dataloader):
            if i >= config["max_steps_per_epoch"]:
                break  # Limiting the number of steps per epoch

            inputs, labels = inputs.to(device), labels.to(device)

            # Using torch.autocast for automatic mixed precision
            with torch.autocast(device_type="cuda"):  # Fixed
                outputs = model(inputs).logits
                input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long, device=device)
                target_lengths = torch.tensor(
                    [len(label[label != -100]) for label in labels], dtype=torch.long, device=device
                )
                loss = criterion(outputs.log_softmax(2).permute(1, 0, 2), labels, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if i % config["log_frequency"] == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        epoch_end_time = time.time()  # Epoch end time
        avg_loss = epoch_loss / config["max_steps_per_epoch"]
        print(f"Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds, Avg Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        epoch_save_path = os.path.join(save_path, f"epoch_{epoch + 1}")
        os.makedirs(epoch_save_path, exist_ok=True)
        model.save_pretrained(epoch_save_path)
        processor.save_pretrained(epoch_save_path)
        print(f"Checkpoint for epoch {epoch + 1} saved to {epoch_save_path}")

        # Log the results of the epoch
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}\n")

    total_end_time = time.time()  # Total time end
    total_time = total_end_time - total_start_time
    print(f"Total training time: {total_time / 3600:.2f} hours")


# ===== Step 5: Main Execution =====
if __name__ == "__main__":
    # Paths
    librispeech_dir = "./LibriSpeech/train-clean-100"  # Path to LibriSpeech directory
    original_tsv = "./train-clean-100.tsv"             # Generated TSV file
    save_directory = "./trained_transformer_model"     # Directory to save the model

    # Training parameters
    config = {
        "batch_size": 10,
        "epochs": 10,
        "learning_rate": 1e-4,
        "max_steps_per_epoch": 550,
        "log_frequency": 10,
    }

    # Create the directory to save the model
    os.makedirs(save_directory, exist_ok=True)
    # Generate the TSV file
    generate_tsv(librispeech_dir, original_tsv)

    # Start training
    train_transformer_model(original_tsv, librispeech_dir, save_directory, config)
