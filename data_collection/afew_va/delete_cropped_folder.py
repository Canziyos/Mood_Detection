import os
import shutil
import stat

DATASET_DIR = r"C:\Users\Dator\OneDrive - Mälardalens universitet\Documents\data\AFEW_VA"
   # The json files were marked as read-only, which caused shutil.rmtree()
    # (permission error), therefore permissions to writable is 
    # used os.chmod(path, stat.S_IWRITE).
def on_rm_error(func, path, exc_info):
    # Remove read-only attribute and try again.
    os.chmod(path, stat.S_IWRITE)
    func(path)

deleted = 0
kept = 0

for folder_name in os.listdir(DATASET_DIR):
    full_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(full_path):
        continue

    if folder_name.endswith("_cropped"):
        print(f"Deleting folder: {folder_name}")
        shutil.rmtree(full_path, onerror=on_rm_error)
        deleted += 1
    else:
        kept += 1

print(f"\nDeleted {deleted} folders.")
print(f"Kept {kept} folders.")
