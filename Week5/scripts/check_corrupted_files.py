from torchvision.io import read_image
import os

def is_valid_jpeg(path):
    try:
        _ = read_image(path)
        return True
    except Exception:
        return False

image_root = 'Week5/dataset/spotting-ball-2023/frames/398x224/england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers'  # <- Set your actual path here
bad_images = []

for root, dirs, files in os.walk(image_root):
    for file in files:
        if file.endswith('.jpg'):
            full_path = os.path.join(root, file)
            if not is_valid_jpeg(full_path):
                bad_images.append(full_path)

print("\n=== Corrupted images ===")
for img in bad_images:
    print(img)

print(f"\nTotal corrupted images: {len(bad_images)}")
