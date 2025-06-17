import os

def create_folders_for_videos():
    """
    Create folders for each video file in the current directory.
    Each folder will have the same name as the video file (without extension).
    """
    # Common video file extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg']
    
    # Get current directory
    directory = os.getcwd()
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter video files
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
    
    if not video_files:
        print("No video files found in the current directory.")
        return
    
    print(f"Found {len(video_files)} video files in '{directory}'.")
    
    created_count = 0
    for video_file in video_files:
        # Get filename without extension
        name_without_ext = os.path.splitext(video_file)[0]
        folder_path = os.path.join(directory, name_without_ext)
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            try:
                os.mkdir(folder_path)
                created_count += 1
                print(f"Created folder: '{name_without_ext}'")
            except OSError as e:
                print(f"Error creating folder '{name_without_ext}': {e}")
        else:
            print(f"Folder '{name_without_ext}' already exists, skipping...")
    
    print(f"\nOperation complete. Created {created_count} new folders out of {len(video_files)} video files.")

if __name__ == "__main__":
    create_folders_for_videos()