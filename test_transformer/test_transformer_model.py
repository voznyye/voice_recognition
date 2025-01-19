import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
from jiwer import wer, cer  # Library for calculating WER and CER
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Load the NLTK library for BLEU (if not installed)
nltk.download('punkt')

# Dataset class for LibriSpeech
class LibriSpeechDataset(Dataset):
    def __init__(self, tsv_path, audio_dir):
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        self.data = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir

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
        # Path to the audio in LibriSpeech (relative, already correct in TSV)
        audio_path = row["path"]  
        # Ensure the path is correct
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        text = row["sentence"]
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        return waveform.squeeze(0), text


class CVCorpusDataset(Dataset):
    def __init__(self, tsv_path, audio_dir):
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        # For CV corpus, it's expected that the file will have 'path' and 'sentence' columns
        self.data = pd.read_csv(tsv_path, sep="\t")
        self.audio_dir = audio_dir

        required_columns = {"path", "sentence"}  # Changed from 'file' to 'path'
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"TSV file must contain columns: {required_columns}")

        self.data = self.data.dropna(subset=["path", "sentence"])
        if len(self.data) == 0:
            raise ValueError(f"No valid data found in {tsv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Remove unnecessary "./" or other artifacts
        audio_path = row["path"].lstrip("./")
        # Path with base directory for CV-Corpus
        audio_path = os.path.join(self.audio_dir, audio_path)
        audio_path = os.path.normpath(audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        text = row["sentence"]
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        return waveform.squeeze(0), text


# ===== Step 3: Data preparation function =====
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


# Test function for the model
def test_model(tsv_path, audio_dir, model_path, processor_path, dataset_type='librispeech', batch_size=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset from: {tsv_path}")
    if dataset_type == 'librispeech':
        test_dataset = LibriSpeechDataset(tsv_path, audio_dir)
    elif dataset_type == 'cv_corpus':
        test_dataset = CVCorpusDataset(tsv_path, audio_dir)
    else:
        raise ValueError("Unsupported dataset type")

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model from: {model_path}")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to(device)

    print(f"Loading processor from: {processor_path}")
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)

            logits = model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Decode the predicted ids
            predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            # Decode true labels
            true_text = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_true_labels.extend(true_text)
            all_predicted_labels.extend(predicted_text)
            
            print(f"True {true_text}")
            print(f"Predict {predicted_text}")

    # Calculate WER, CER, and BLEU
    wer_score, cer_score, bleu_score = calculate_speech_metrics(all_true_labels, all_predicted_labels)

    print(f"Word Error Rate (WER): {wer_score:.4f}")
    print(f"Character Error Rate (CER): {cer_score:.4f}")
    print(f"BLEU Score: {bleu_score:.4f}")


# Function to calculate WER, CER, and BLEU
def calculate_speech_metrics(true_labels, predicted_labels):
    wer_score = wer(true_labels, predicted_labels)
    cer_score = cer(true_labels, predicted_labels)
    
    bleu_scores = []
    for true, pred in zip(true_labels, predicted_labels):
        # For BLEU, we treat each label pair as a single hypothesis/reference
        references = [true.split()]
        hypothesis = pred.split()
        bleu_scores.append(sentence_bleu(references, hypothesis))
    
    bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    return wer_score, cer_score, bleu_score


# Testing the model
if __name__ == "__main__":    
    audio_directory = "./cv-corpus-20.0-delta-2024-12-06/en/clips"  # Path to the CV corpus audio files
    model_path = "./trained_transformer_model/epoch_10"  # Path to the trained model
    processor_path = "./trained_transformer_model/epoch_10"  # Path to the processor

    # Paths for the test data
    test_dataset_paths = {
        "librispeech": "./test-other.tsv",
        "cv_corpus": "./cv-corpus-20.0-delta-2024-12-06/en/validated.tsv"
    }

    # Directories for audio
    audio_directories = {
        "librispeech": "./LibriSpeech/train-clean-100",  # LibriSpeech uses absolute paths
        "cv_corpus": "./cv-corpus-20.0-delta-2024-12-06/en/clips"
    }

    # Test both datasets
    for dataset_type, tsv_path in test_dataset_paths.items():
        test_model(
            tsv_path=tsv_path,
            audio_dir=audio_directories[dataset_type],
            model_path=model_path,
            processor_path=processor_path,
            dataset_type=dataset_type
        )
