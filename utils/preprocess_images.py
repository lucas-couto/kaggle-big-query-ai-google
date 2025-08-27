import os
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

def preprocess_images(config):
    # --- config ---
    input_shape = config['model']['input_shape']          # ex.: [224, 224, 3]
    image_size = (input_shape[0], input_shape[1])
    input_dir = Path(config['paths']['images_dir'])
    output_dir = Path(config['paths']['datasets_dir'])

    # preferir 'test_size' se existir; senão, usar 1 - train_size ou fallback para 0.2
    train_size = float(config['training'].get('train_size', 0.8))
    test_size = float(config['training'].get('test_size', 1.0 - train_size))
    random_state = int(config['training'].get('random_state', 42))

    # extensões aceitas
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

    # --- reset do output ---
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)

    # --- coletar imagens e rótulos ---
    # cada subpasta imediata de input_dir é uma classe
    classes = sorted([p.name for p in input_dir.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"Nenhuma subpasta/classe encontrada em {input_dir}")

    samples = []  # lista de caminhos relativos (classe/arquivo)
    labels  = []  # lista com o nome da classe correspondente

    for cls in classes:
        cls_dir = input_dir / cls
        for f in cls_dir.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                rel = Path(cls) / f.name
                samples.append(rel)
                labels.append(cls)

    if not samples:
        raise RuntimeError(f"Nenhuma imagem válida encontrada em {input_dir} (extensões: {sorted(exts)})")

    # --- split estratificado train/test ---
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = list(range(len(samples)))
    train_idx, test_idx = next(splitter.split(indices, labels))

    def _save_split(idxs, split_name):
        for i in idxs:
            rel = samples[i]
            cls = labels[i]
            src = input_dir / rel
            dst = output_dir / split_name / cls / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(src) as im:
                    # converte para RGB se necessário (evita problemas com PNG/alpha, P, L etc.)
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    im = im.resize(image_size)
                    im.save(dst)
            except Exception as e:
                print(f"[WARN] Erro ao processar {src}: {e}")

    _save_split(train_idx, 'train')
    _save_split(test_idx,  'test')

    # relatório rápido
    print("Classes:", classes)
    print(f"Total: {len(samples)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    # contagem por classe
    from collections import Counter
    tr_counts = Counter([labels[i] for i in train_idx])
    te_counts = Counter([labels[i] for i in test_idx])
    print("Train por classe:", dict(tr_counts))
    print("Test por classe:", dict(te_counts))
