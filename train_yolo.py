from ultralytics import YOLO

# Carregar modelo treinado
model = YOLO("runs/detect/runs/train/cube_detection2/weights/best.pt")

# Prever uma imagem
img_path = "rubik_dataset/images/val/6f5208c5b6943b6c97db4c6f557aba19.jpg"  # Altere para o caminho desejado
results = model.predict(source=img_path)
results.show()  # mostra a imagem com bounding boxes
