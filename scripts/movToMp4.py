import os
from moviepy import VideoFileClip
from tqdm import tqdm

input_folder = "C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\video_snippets"
output_folder = "C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\mp4_snippets"

mov_files = [filename for filename in os.listdir(input_folder) if filename.endswith(".mov")]

for filename in tqdm(mov_files, desc='Converting videos', unit='video'):
    input_file_path = os.path.join(input_folder, filename)
    video = VideoFileClip(input_file_path)
    
    duration = video.duration
    total_frames = int(duration * video.fps)
    
    output_filename = os.path.splitext(filename)[0] + ".mp4"
    output_file_path = os.path.join(output_folder, output_filename)
    
    video.write_videofile(output_file_path, codec='libx264', fps=video.fps)
    
    progress = 0
    with tqdm(total=total_frames, desc=f'{filename}', unit='frame') as pbar:
        while progress < total_frames:
            progress = video.reader.nframes
            pbar.update(progress - pbar.n)