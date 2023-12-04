import cv2
import dlib
import numpy as np
import time

# Inicializa o rastreador
def init_tracker(frame, bbox):
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    tracker.start_track(frame, rect)
    return tracker

# Calibra as coordenadas do vídeo
def calibrate_coordinates(video_path, real_coordinates, video_coordinates):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # Mapeamento de coordenadas reais para coordenadas do vídeo
    M = cv2.getPerspectiveTransform(np.float32(real_coordinates), np.float32(video_coordinates))

    return M

# Rastreia o jogador selecionado sem exibir a trajetória
def track_selected_player(video_path, calibration_matrix):
    cap = cv2.VideoCapture(video_path)

    # Encontrar manualmente as coordenadas do jogador inicial
    ret, frame = cap.read()
    bbox = cv2.selectROI('Select Player', frame, False)
    cv2.destroyWindow('Select Player')

    # Inicializar o rastreador para o jogador selecionado
    tracker = init_tracker(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rastrear o jogador
        tracker.update(frame)
        rect = tracker.get_position()
        p1 = (int(rect.left()), int(rect.top()))
        p2 = (int(rect.right()), int(rect.bottom()))

        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

        cv2.imshow('Selected Player Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "video.mp4"

    # Coordenadas reais e correspondentes no vídeo (calibre conforme necessário)
    real_coordinates = np.float32([(0, 0), (10, 0), (10, 10), (0, 10)])
    video_coordinates = np.float32([(100, 200), (500, 200), (500, 400), (100, 400)])

    # Obtém a matriz de calibração
    calibration_matrix = calibrate_coordinates(video_path, real_coordinates, video_coordinates)

    # Rastreia o jogador selecionado sem exibir a trajetória
    track_selected_player(video_path, calibration_matrix)
