import pandas as pd
import librosa
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import pickle
from torch import nn
from torchvision.models import efficientnet_b0
import os

if os.path.exists('/kaggle/input'):
    ROOT = '/kaggle/input/birdclef-2025/'
    DURATIONS_FILE = '/kaggle/input/bird-clef-utils/durations.csv'
    LABEL_MAP_FILE = '/kaggle/input/bird-clef-utils/label_map.json'
    VOICE_SEGMENTS_FILE = '/kaggle/input/bird-clef-utils/train_voice_data.pkl'
    MODEL_OUTPUT_PATH = '/kaggle/working/my_model.pth'
else:
    ROOT = 'C:/Users/aschieb/Desktop/birdclef-2025/'
    DURATIONS_FILE = ROOT + 'durations.csv'
    LABEL_MAP_FILE = ROOT + 'label_map.json'
    VOICE_SEGMENTS_FILE = ROOT + 'train_voice_data.pkl'
    MODEL_OUTPUT_PATH = ROOT + 'my_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segment_length = 5  # seconds
stride = 5  # or 1


class EffNetB0Classifier(nn.Module):
    def __init__(self, num_classes=206):
        super(EffNetB0Classifier, self).__init__()
        # Load the pre-trained EfficientNet B0 model
        self.effnet = efficientnet_b0()
        # Remove the original classifier
        self.features = self.effnet.features
        self.pooling = self.effnet.avgpool
        self.flatten = nn.Flatten()
        
        # Define a custom MLP with hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(self.effnet.classifier[1].in_features, 512),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x


def process_spectrogram(spectrogram_db):
    spectrogram_tensor = torch.tensor(spectrogram_db, dtype=torch.float32).unsqueeze(0)
    spectrogram_tensor = torch.nn.functional.interpolate(spectrogram_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    spectrogram_tensor = spectrogram_tensor.squeeze(0).repeat(3, 1, 1).to(device)
    return spectrogram_tensor


class AudioDataset(Dataset):
    def __init__(self, training_chunks):
        self.training_chunks = training_chunks
        self.label_to_int = json.load(open(LABEL_MAP_FILE))

    def __len__(self):
        return len(self.training_chunks)

    def __getitem__(self, idx):
        row = self.training_chunks.iloc[idx]
        audio_path = ROOT + 'train_audio/' + row['filename']
        audio_start = row['audio_start']
        primary_label = row['primary_label']

        # Load the audio file
        waveform, sample_rate = librosa.load(audio_path, sr=None, offset=audio_start, duration=5.0)

        # If the audio is less than 5 seconds, pad with zeros
        if len(waveform) < 5 * sample_rate:
            padding = 5 * sample_rate - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')

        # Compute the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_tensor = process_spectrogram(spectrogram_db)

        # Map primary_label to an integer
        primary_label = self.label_to_int[primary_label]

        # Return x (spectrogram) and y (primary_label)
        return spectrogram_tensor, primary_label



def load_all_training_chunks():
    if os.path.exists('training_chunks.csv'):
        return pd.read_csv('training_chunks.csv')

    # Load the train.csv file
    df = pd.read_csv(ROOT+'train.csv')
    durations = pd.read_csv(DURATIONS_FILE)


    # Merge df with durations on filename
    df = df.merge(durations, on="filename")
    df["file_length"] = df["file_length"].astype(float)

    # Create a new dataframe for training chunks
    training_chunks = []

    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        file_length = row["file_length"]
        num_chunks = int(file_length // segment_length)
        for start in range(0, num_chunks * segment_length, stride):
            chunk = row.copy()
            chunk["audio_start"] = start
            training_chunks.append(chunk)
        # Add the remaining chunk if any
        if file_length % segment_length > 0:
            chunk = row.copy()
            chunk["audio_start"] = num_chunks * segment_length
            training_chunks.append(chunk)

    # Convert the list of chunks into a dataframe
    training_chunks = pd.DataFrame(training_chunks)
    training_chunks.drop(columns=['url','scientific_name','common_name','author','collection','author','license'], inplace=True)
    return training_chunks



# Remove human voice
with open(VOICE_SEGMENTS_FILE, 'rb') as file:
    train_voice_data = pickle.load(file)

def contains_human_voice(filename, audio_start, audio_end):
    #TODO: adapt this for train soundscapes too

    # quick hack because original author used the kaggle path
    filename = "/kaggle/input/birdclef-2025/train_audio/" + filename

    if filename not in train_voice_data:
        return False

    for segment in train_voice_data[filename]:
        if segment['start'] <= audio_end and segment['end'] >= audio_start:
            return True

    return False

