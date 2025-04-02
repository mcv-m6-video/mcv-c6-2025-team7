from PIL import Image
import os

def fix_jpeg(path):
    try:
        img = Image.open(path)
        img.save(path, 'JPEG')
    except Exception as e:
        print(f"Failed to fix {path}: {e}")

# Walk through and fix images
for root, dirs, files in os.walk('Week5/dataset/spotting-ball-2023/frames/398x224/england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest'):
    for f in files:
        if f.endswith('.jpg'):
            fix_jpeg(os.path.join(root, f))
