from pathlib import Path

from PIL import Image

# Cargar el video
images_path = Path(r'C:\Users\gerar\Documents\MCV\C6\MCV_C6_G7_2025\Week1\Output_Videos\AdaptativeModelling\videos_for_gif')

# Crear una lista para almacenar los frames del video
frames = []

# Leer el video frame por frame
frame_count = 0
for img_file in images_path.glob('*.png'):
    frames.append(Image.open(img_file))

# Guardar el gif
gif_path = r"C:\Users\gerar\Documents\MCV\C6\MCV_C6_G7_2025\Week1\Output_Videos\AdaptativeModelling\estimated_background.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=1000/1, loop=0)

print(f"GIF guardado como {gif_path}")
