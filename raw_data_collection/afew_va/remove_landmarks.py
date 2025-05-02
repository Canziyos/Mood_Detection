import os
import json

ROOT_DIR = r"Replace_with_your_unzipped_folder\AFEW_VA"

count = 0
for i in range(1, 601):
    folder_name = f"{i:03d}"
    folder_path = os.path.join(ROOT_DIR, folder_name)
    json_path = os.path.join(folder_path, f"{folder_name}.json")

    if not os.path.exists(json_path):
        print(f"Missing JSON: {json_path}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    if "frames" not in data:
        print(f"No 'frames' key in {json_path}")
        continue

    # Only remove 'landmarks' key from each frame, keep all else.
    for frame_id in data["frames"]:
        if "landmarks" in data["frames"][frame_id]:
            del data["frames"][frame_id]["landmarks"]

    # Save it back.
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    count += 1

print(f"\nCleaned 'landmarks' from {count} JSON files.")
