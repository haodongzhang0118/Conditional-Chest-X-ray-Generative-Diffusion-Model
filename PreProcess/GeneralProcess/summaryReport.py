from transformers import pipeline
import os
from tqdm import tqdm

summarizer = pipeline("summarization", model="Mbilal755/Radiology_Bart")
path = r"D:\UCI_documents\2023_Fall_Quarter\CS 184a\FinalProject\files"

index = 0

for subfolder in os.listdir(path):
    for sub_subfolder in tqdm(os.listdir(os.path.join(path, subfolder)), desc="Processing {}: ".format(index + 1)):
        for sub_sub_subfolder in os.listdir(os.path.join(path, subfolder, sub_subfolder)):
                path_txt = os.path.join(path, subfolder, sub_subfolder, sub_sub_subfolder, "label")
                if "summary.txt" not in os.listdir(path_txt):
                    for file in os.listdir(path_txt):
                        with open(os.path.join(path_txt, file), 'r') as f:
                            findings = f.read()
                            summary = summarizer(findings, max_length=105)[0]['summary_text']
                            with open(os.path.join(path_txt, "summary.txt"), 'w') as f:
                                f.write(summary)
    index += 1