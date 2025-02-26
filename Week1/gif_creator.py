import cv2
from PIL import Image

# Cargar el video
video_path = 'Output_Videos/task_1_1_mean_alpha10.avi'
cap = cv2.VideoCapture(video_path)

# Obtener el fps (frames por segundo) y el tamaño de los frames
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear una lista para almacenar los frames del video
frames = []

# Leer el video frame por frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count > fps * 5:  # Limitar a los primeros 5 segundos
        break
    # Convertir el frame de BGR a RGB (ya que Pillow usa RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(rgb_frame))
    frame_count += 1

# Cerrar el video
cap.release()

# Guardar el gif
gif_path = "video_convertido.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=1000/fps, loop=0)

print(f"GIF guardado como {gif_path}")
