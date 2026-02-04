import os
import glob
import random
import shutil

# Diretórios
base_dir = 'rubik_dataset'
img_dir = os.path.join(base_dir, 'images')
lbl_dir = os.path.join(base_dir, 'labels')

train_img_dir = os.path.join(img_dir, 'train')
val_img_dir = os.path.join(img_dir, 'val')
train_lbl_dir = os.path.join(lbl_dir, 'train')
val_lbl_dir = os.path.join(lbl_dir, 'val')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# Lista de imagens
img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
img_files.sort()

# Embaralha e divide
random.seed(42)
random.shuffle(img_files)
n_total = len(img_files)
n_train = int(n_total * 0.8)
train_imgs = img_files[:n_train]
val_imgs = img_files[n_train:]

def move_files(img_list, dest_img_dir, dest_lbl_dir):
    for img_path in img_list:
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
        # Move imagem
        shutil.move(img_path, os.path.join(dest_img_dir, fname))
        # Move label
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(dest_lbl_dir, fname.replace('.jpg', '.txt')))
        else:
            print(f'Label não encontrada para {fname}')

move_files(train_imgs, train_img_dir, train_lbl_dir)
move_files(val_imgs, val_img_dir, val_lbl_dir)


# Mover imagens e labels restantes para train se tiverem .txt
remaining_imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
for img_path in remaining_imgs:
    fname = os.path.basename(img_path)
    lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
    if os.path.exists(lbl_path):
        shutil.move(img_path, os.path.join(train_img_dir, fname))
        shutil.move(lbl_path, os.path.join(train_lbl_dir, fname.replace('.jpg', '.txt')))
        print(f'Movido para train: {fname}')
    else:
        print(f'Ignorado (sem label): {fname}')

print(f'Treino: {len(os.listdir(train_img_dir))} imagens')
print(f'Validação: {len(os.listdir(val_img_dir))} imagens')
print('Divisão concluída!')
