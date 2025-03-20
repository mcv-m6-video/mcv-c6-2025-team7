import cv2

# Video details
sequence = "s03"
cases = ["c010", "c011", "c012", "c013", "c014", "c015"]  # Cameras
cases = ["c014"]  # Cameras

def change_fps(input_file, output_file, target_fps):
    cap = cv2.VideoCapture(input_file)

    # Obtener propiedades del video original
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Usa el códec XVID para AVI
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"FPS original: {original_fps}, FPS objetivo: {target_fps}")

    # Crear el nuevo video con el FPS deseado
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video guardado en {output_file}")

# Target FPS (standardize frame rate for all videos)
target_fps = 10  # Desired frame rate (e.g., 30 FPS)

# Loop through all cases and process their videos
for case in cases:
    input_video = f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi"
    output_video = f"videos/{sequence}_{case}.avi"  # CFR output

    # Convertir un video a 30 FPS
    change_fps(input_video, output_video, target_fps)

    