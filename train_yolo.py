from ultralytics import YOLO
import cv2

# Treinamento do modelo YOLOv8
model = YOLO('yolov8s.pt')
results = model.train(
    data='rubik_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs/train',
    name='cube_detection'
)

img_path = 'rubik_dataset/images/val/cube_000.jpg'
pred = model(img_path)
pred[0].show()  # Exibe a imagem com as detecções

pred[0].save('predicted.jpg')
print('Previsão salva em predicted.jpg')
