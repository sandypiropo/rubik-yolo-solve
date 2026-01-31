# Cubo Mágico - Detecção de Estado com YOLO
# Tecnologias: Python, OpenCV, YOLOv8, Numpy
# Objetivo: Detectar automaticamente um Cubo Mágico usando YOLO e verificar se ele está montado, analisando cores das faces visíveis.

import cv2
import numpy as np
from ultralytics import YOLO

# Função para capturar vídeo ao vivo
def capture_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Não foi possível acessar a webcam.")
    return cap

# Função para detectar cubo com YOLO
def detect_cube_yolo(model, frame):
    results = model(frame)
    boxes = []
    # Defina o índice da classe do cubo mágico conforme seu modelo customizado
    CUBE_CLASS_IDX = 0  # Altere para o índice correto do cubo mágico no seu modelo
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == CUBE_CLASS_IDX:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
    return boxes
# Função para processar cores e dividir em 9 quadrados
def process_cube_face(frame, bbox):
    x1, y1, x2, y2 = bbox
    cube_img = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cube_img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    grid = []
    step_x = w // 3
    step_y = h // 3
    for row in range(3):
        for col in range(3):
            x_start = col * step_x
            y_start = row * step_y
            x_end = (col + 1) * step_x if col < 2 else w
            y_end = (row + 1) * step_y if row < 2 else h
            square = hsv[y_start:y_end, x_start:x_end]
            # Média da cor HSV do quadrado
            mean_hsv = cv2.mean(square)[:3]
            grid.append(mean_hsv)
    return grid  # Lista de 9 médias HSV
# Função para classificar estado
def classify_cube_state(hsv_grid, tol=20):
    # Considera 'Montado' se todos os quadrados centrais têm cor próxima
    # Simples: verifica se há 6 cores distintas (centro de cada face)
    # Aqui, como só vemos uma face, considera montado se todos os 9 quadrados são próximos do centro
    center_color = hsv_grid[4]
    for hsv in hsv_grid:
        if np.linalg.norm(np.array(hsv) - np.array(center_color)) > tol:
            return 'Não montado'
    return 'Montado'
# Função principal para orquestrar o fluxo

def main():
    # Carregar modelo YOLOv8 pré-treinado (substitua por caminho do seu modelo customizado se necessário)
    model = YOLO('yolov8n.pt')  # Use um modelo customizado para cubo mágico se disponível
    cap = capture_video()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break
        # Detecção do cubo
        boxes = detect_cube_yolo(model, frame)
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            # Processar face do cubo
            hsv_grid = process_cube_face(frame, (x1, y1, x2, y2))
            status = classify_cube_state(hsv_grid)
            # Exibir bounding box e status
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if status=='Não montado' else (0,255,0), 2)
        cv2.imshow('RubikVisionSolve - Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
