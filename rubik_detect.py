import cv2
import numpy as np
from ultralytics import YOLO
import datetime

# --- Configurações globais ---
LOG_FILE = 'rubikvision.log'
MODEL_PATH = 'runs/detect/runs/train/cube_detection2/weights/best.pt'
CUBE_CLASS_IDX = 0
COLOR_TOLERANCE = 80

def log_event(message: str):
    """Registra mensagem no arquivo de log com timestamp."""
    with open(LOG_FILE, 'a', encoding='utf-8') as logf:
        logf.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        logf.flush()

def capture_video():
    """Inicializa a captura de vídeo da webcam padrão."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Não foi possível acessar a webcam.")
    return cap

def detect_cube_yolo(model, frame):
    """Detecta cubos mágicos na imagem usando YOLO."""
    results = model(frame)
    boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == CUBE_CLASS_IDX:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                boxes.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
    return boxes

def process_cube_face(frame, bbox):
    """Divide a face do cubo em 9 quadrados e retorna a média HSV de cada um."""
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
            mean_hsv = cv2.mean(square)[:3]
            grid.append(mean_hsv)
    return grid

def classify_cube_state(hsv_grid, tol=COLOR_TOLERANCE):
    """
    Classifica o estado do cubo como montado ou não montado baseado na similaridade das cores.
    Otimizado para tolerar pequenas variações e adaptar-se a diferentes condições de iluminação.
    Considera montado se pelo menos 7 de 9 quadrados forem próximos do centro.
    Também faz uma checagem robusta para outliers e adapta a tolerância se houver muita variação global.
    """
    center_color = np.array(hsv_grid[4])
    diffs = [np.linalg.norm(np.array(hsv) - center_color) for hsv in hsv_grid]
    # Adaptação: se a variância das diferenças for alta, aumenta a tolerância
    var = np.var(diffs)
    adaptive_tol = tol + min(var, 40)  # Limita o aumento para não ser exagerado
    # Considera montado se pelo menos 7 quadrados são próximos do centro
    close_count = sum(d < adaptive_tol for d in diffs)
    if close_count >= 7:
        return 'Cubo Montado'
    return 'Cubo Não Montado'

def main():
    """Função principal: inicializa, executa detecção e logging, e exibe resultados."""
    # Limpa o arquivo de log no início de cada execução
    with open(LOG_FILE, 'w', encoding='utf-8') as logf:
        logf.write('')

    model = YOLO(MODEL_PATH)
    cap = capture_video()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_event('Falha ao capturar frame.')
                break
            boxes = detect_cube_yolo(model, frame)
            for box in boxes:
                x1, y1, x2, y2 = box['bbox']
                hsv_grid = process_cube_face(frame, (x1, y1, x2, y2))
                status = classify_cube_state(hsv_grid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                if status == 'Cubo Montado':
                    label = 'Rubik Solved'
                    color = (0,255,0)
                else:
                    label = 'Rubik Not Solved'
                    color = (0,0,255)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                print(label)
                log_event(f"{label}: bbox=({x1}, {y1}, {x2}, {y2})")
            cv2.imshow('RubikVisionSolve - Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
