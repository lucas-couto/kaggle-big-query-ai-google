import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

def preprocess_images(config):
    input_shape = config['model']['input_shape']
    image_size = (input_shape[0], input_shape[1])
    input_dir = Path(config['paths']['images_dir'])
    output_dir = Path(config['paths']['datasets_dir'])

    train_size = float(config['data'].get('train_size', 0.8))
    random_state = int(config['training'].get('random_state', 42))

    # Limpa pasta de saída
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)

    # Coleta imagens e classes
    samples, labels = [], []
    for cls in sorted(os.listdir(input_dir)):
        cls_path = input_dir / cls
        if cls_path.is_dir():
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    samples.append(Path(cls) / file)
                    labels.append(cls)

    num_classes = len(set(labels))
    test_size = max(int(len(samples) * (1 - train_size)), num_classes)
    if test_size >= len(samples):
        test_size = num_classes  # garantir pelo menos 1 por classe

    # Split estratificado
    train_imgs, test_imgs = train_test_split(
        samples, test_size=test_size, stratify=labels, random_state=random_state
    )

    def save_images(img_list, split):
        for img in img_list:
            src = input_dir / img
            dst = output_dir / split / img
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB").resize(image_size)
                im.save(dst)

    save_images(train_imgs, 'train')
    save_images(test_imgs, 'test')

    print(f"Total de imagens: {len(samples)}")
    print(f"Treino: {len(train_imgs)} | Teste: {len(test_imgs)}")
