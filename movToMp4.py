import os
from moviepy import VideoFileClip
from tqdm import tqdm

# code to convert taken from medium - nam nguyen

input_folder = "C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\video_snippets"  # Specify the input folder containing .mov files
output_folder = "C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\mp4_snippets"  # Specify the output folder for .mp4 files# Get a list of all .mov files in the input folder
mov_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".mov")]# Loop through all .mov files and show estimated progress with a progress bar
for filename in tqdm(mov_files, desc='Converting videos', unit='video'):
    # Get the full file path of the input .mov file
    input_file_path = os.path.join(input_folder, filename)
    
    # Load the .mov file
    video = VideoFileClip(input_file_path)
    
    # Get the duration of the video in seconds
    duration = video.duration
    
    # Estimate the total number of frames to be written based on duration and frame rate
    total_frames = int(duration * video.fps)
    
    # Construct the output file name with .mp4 extension
    output_filename = os.path.splitext(filename)[0] + ".mp4"
    
    # Construct the output file path
    output_file_path = os.path.join(output_folder, output_filename)
    
    # Convert and save the video to .mp4 in the output folder
    video.write_videofile(output_file_path, codec='libx264', fps=video.fps)
    
    # Update the progress manually based on the progress of video writing
    progress = 0
    with tqdm(total=total_frames, desc=f'{filename}', unit='frame') as pbar:
        while progress < total_frames:
            # Update progress based on the number of frames written
            progress = video.reader.nframes
            pbar.update(progress - pbar.n)