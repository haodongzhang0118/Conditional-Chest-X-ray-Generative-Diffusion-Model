import pandas as pd
import numpy as np
import random

# Load the CSV file
df = pd.read_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\modified_file.csv")

# Columns to modify
columns_to_modify = list(range(2, 16))  # Adjust this range as per your columns

# Iterate through each row
for index, row in df.iterrows():
    # Find columns with value 1
    cols_with_1 = [col for col in columns_to_modify if row[col] == 1]

    # If more than one column has 1, randomly select one and set others to NaN
    if len(cols_with_1) > 1:
        if 10 in cols_with_1:
            cols_with_1.remove(10)
            df.iloc[index, 10] = pd.NA
        selected_col = random.choice(cols_with_1)
        for col in cols_with_1:
            if col != selected_col:
                df.iloc[index, col] = pd.NA

# Save the modified DataFrame to a new CSV file
df.to_csv(r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\Useful Files\modified_file_with_single_1.csv", index=False)