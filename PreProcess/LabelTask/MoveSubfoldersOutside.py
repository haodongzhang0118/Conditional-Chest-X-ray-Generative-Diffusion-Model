import os
import shutil

def move_subsubfolders_to_main(main_folder):
    # Iterate over all subdirectories in the main folder
    for subfolder in [f.path for f in os.scandir(main_folder) if f.is_dir()]:
        
        # Iterate over all subsubdirectories in each subfolder
        for subsubfolder in [f.path for f in os.scandir(subfolder) if f.is_dir()]:
            
            # Construct the new path in the main folder
            new_path = os.path.join(main_folder, os.path.basename(subsubfolder))
            
            # Move the subsubfolder to the main folder
            shutil.move(subsubfolder, new_path)
        
        # Attempt to remove the now-empty subfolder
        try:
            os.rmdir(subfolder)
        except OSError:
            print(f"Warning: Subfolder {subfolder} is not empty and was not removed.")

# Replace 'path/to/main_folder' with the path to your main folder
main_folder_path = r"D:\MIMIC_Preprocessed_Data"
move_subsubfolders_to_main(main_folder_path)
