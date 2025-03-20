import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path


# Load the video
parser = argparse.ArgumentParser(description="Object tracking script")
parser.add_argument("--sequence", required=True, help="Nombre de la secuencia (ej. S01)")
parser.add_argument("--case", required=True, help="Nombre del caso (ej. c002)")
parser.add_argument("--parked", type=bool, default=False, help="Si es True, elimina los coches estacionados, si es False, los considera")

args = parser.parse_args()
sequence = args.sequence.lower()  # Convertimos a minúsculas por consistencia
case = args.case.lower()
parked = args.parked  
seq_case_name = f"{sequence}_{case}"

output_dir = Path(f"TrackEval/data/trackers/mot_challenge/{seq_case_name}-train/PerfectTracker/data")

# if parked:
#     video_path = output_dir / f"{seq_case_name}-01_parked.avi"
# else:
#     video_path = output_dir / f"{seq_case_name}-01.avi"

video_path = f"output_{sequence}_{case}_mod.avi"
    
cap = cv2.VideoCapture(video_path)

# Get FPS and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Reduce frame size to 50% of original
scale_percent = 50
new_width = int(frame_width * scale_percent / 100)
new_height = int(frame_height * scale_percent / 100)

# Frames list
frames = []

# Capture only 8 seconds from frame 500
frames_ini = 100
frames_a_capturar = int(fps * 20)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    if frame_count < frames_ini:  
        frame_count += 1
        continue  # Skip frames before 500

    if frame_count >= frames_ini + frames_a_capturar:
        break  # Stop after 8 seconds

    if frame_count % 2 == 0:  # Skip every other frame (reduce FPS)
        # Resize frame to 50%
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to Pillow Image
        img = Image.fromarray(rgb_frame)
        
        # Reduce color palette to 128 colors
        img = img.convert('P', palette=Image.ADAPTIVE, colors=128)
        
        frames.append(img)

    frame_count += 1

# Close video
cap.release()

# Save optimized GIF
gif_path = f"gif_{sequence}_{case}.gif"
if frames:
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=1000/fps, loop=0)
    print(f"Optimized GIF saved as {gif_path}")
else:
    print("No frames were generated for the GIF.")
