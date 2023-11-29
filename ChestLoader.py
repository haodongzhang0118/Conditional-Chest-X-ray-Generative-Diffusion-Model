import os
from torch.utils.data import Dataset
from PIL import Image
from transformers import (AutoConfig, AutoTokenizer, AutoModel)
from tqdm import tqdm
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()
        config = AutoConfig.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.tokenizer = AutoTokenizer.from_pretrained('zzxslp/RadBERT-RoBERTa-4m')
        self.model = AutoModel.from_pretrained('zzxslp/RadBERT-RoBERTa-4m', config=config)

    def _load_samples(self):
        samples = []
        index = 0
        for p_dir in os.listdir(self.root_dir):
            p_path = os.path.join(self.root_dir, p_dir)
            if os.path.isdir(p_path):
                for subject in tqdm(os.listdir(p_path), desc="Processing data from {}".format(index + 1)):
                    subject_path = os.path.join(p_path, subject)
                    for study in os.listdir(subject_path):
                        study_path = os.path.join(subject_path, study)
                        image_path = os.path.join(study_path, "images")
                        label_path = os.path.join(study_path, "label")
                        image_file = os.listdir(image_path)[0]
                        label_file = "summary.txt"
                        samples.append((os.path.join(image_path, image_file),
                                        os.path.join(label_path, label_file)))
            index += 1
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        with open(label_path, 'r') as file:
            label = file.read()

        if self.transform:
            image = self.transform(image)
        
        processed_label = self.process_report(label)

        # image size: (256, 3, 256, 256)
        # label size: (256, 1, 64, 768)

        return image, processed_label
    
    def process_report(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
        with torch.no_grad():
            processed_text = self.model(**encoded_input)["last_hidden_state"] # (batch_size, sequence_length, hidden_size)
        return processed_text
