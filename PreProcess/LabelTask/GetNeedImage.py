import pandas as pd
import os
from tqdm import tqdm

subject_id = []
study_id = []

main_folder = r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\MIMIC_Preprocessed_Data"

index = 1
for folder in os.listdir(main_folder):
    subject_id += os.listdir(os.path.join(main_folder, folder))
    for sub_folder in tqdm(os.listdir(os.path.join(main_folder, folder)), desc="Processing {}: ".format(index)):
        study_id += os.listdir(os.path.join(main_folder, folder, sub_folder))
    index += 1

# Load the CSV file
df = pd.read_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\final.csv")

# Filter condition for subject_id
subject_id_condition = df['subject_id'].isin(subject_id)

# Filter condition for study_id
study_id_condition = df['study_id'].isin(study_id)

# Apply the filter conditions
filtered_df = df[subject_id_condition & study_id_condition]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\filtered_file.csv", index=False)
