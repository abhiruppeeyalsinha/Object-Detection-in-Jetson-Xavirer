import os

# Set the path to your target folder
folder_path = '/path/to/your/folder'

# Common video file extensions
video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv'}

# Ensure the directory exists
if os.path.isdir(folder_path):
    for item in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item)

        if os.path.isfile(file_path):
            name, ext = os.path.splitext(item)

            if ext.lower() in video_extensions:
                new_folder_path = os.path.join(folder_path, name)

                # Create a folder if it doesn't already exist
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path)
                    print(f"Created folder: {new_folder_path}")
else:
    print("Invalid folder path. Please check again.")
