import pandas as pd
import numpy as np
import random

subject_id_need = []
study_id_need = []
subject_id_not_need = []
study_id_not_need = []
labels = []

# Load the CSV file
df = pd.read_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\filtered_file.csv")

# Columns to modify
columns_to_modify = list(range(2, 16))  # Adjust this range as per your columns

num = 0
# Iterate through each row
for index, row in df.iterrows():
    # Find columns with value 1
    cols_with_1 = [col for col in columns_to_modify if row[col] == 1]

    if len(cols_with_1) == 0 or len(cols_with_1) > 1:
        subject_id_not_need.append(row[0])
        study_id_not_need.append(row[1])
    else:
        subject_id_need.append(row[0])
        study_id_need.append(row[1])
        labels.append(cols_with_1[0] - 2)

subject_id_need_condition = df['subject_id'].isin(subject_id_need)
study_id_need_condition = df['study_id'].isin(study_id_need)
subject_id_noneed_condition = df['subject_id'].isin(subject_id_not_need)
study_id_noneed_condition = df['study_id'].isin(study_id_not_need)

df_need = df[subject_id_need_condition & study_id_need_condition]
df_noneed = df[subject_id_noneed_condition & study_id_noneed_condition]

df_need['label'] = labels

df_need.to_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\filtered_file_need.csv", index=False)
df_noneed.to_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\filtered_file_noneed.csv", index=False)

