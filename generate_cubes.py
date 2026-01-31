
import cv2
import numpy as np
import os
import random

# Cores típicas do cubo mágico (BGR)
CUBE_COLORS = [
    (0, 255, 255),   # Amarelo
    (0, 140, 255),   # Laranja
    (0, 0, 255),     # Vermelho
    (0, 255, 0),     # Verde
    (255, 0, 0),     # Azul
    (255, 255, 255)  # Branco
]

IMG_SIZE = 256
CUBE_SIZE = 180
MARGIN = (IMG_SIZE - CUBE_SIZE) // 2

os.makedirs('dataset/montado', exist_ok=True)
os.makedirs('dataset/nao_montado', exist_ok=True)

# Gerar 6 imagens montadas (uma para cada cor do cubo)
for i, cor in enumerate(CUBE_COLORS):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 50  # fundo cinza escuro
    for row in range(3):
        for col in range(3):
            x0 = MARGIN + col * (CUBE_SIZE // 3)
            y0 = MARGIN + row * (CUBE_SIZE // 3)
            x1 = x0 + (CUBE_SIZE // 3) - 4
            y1 = y0 + (CUBE_SIZE // 3) - 4
            cv2.rectangle(img, (x0, y0), (x1, y1), cor, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0,0,0), 2)
    cv2.imwrite(f'dataset/montado/cube_{i:03d}.png', img)

# Gerar 50 imagens não montadas (cores variadas)
for i in range(50):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 50
    face_colors = random.choices(CUBE_COLORS, k=9)
    while len(set(face_colors)) == 1:
        face_colors = random.choices(CUBE_COLORS, k=9)  # Garante que não seja igual ao montado
    for row in range(3):
        for col in range(3):
            x0 = MARGIN + col * (CUBE_SIZE // 3)
            y0 = MARGIN + row * (CUBE_SIZE // 3)
            x1 = x0 + (CUBE_SIZE // 3) - 4
            y1 = y0 + (CUBE_SIZE // 3) - 4
            color = face_colors[row * 3 + col]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0,0,0), 2)
    cv2.imwrite(f'dataset/nao_montado/cube_{i:03d}.png', img)

print('Imagens geradas em ./dataset/montado e ./dataset/nao_montado')
