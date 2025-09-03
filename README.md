# Sistema de Classificação de Imagens com CNN e Modelos Pré-treinados

Este projeto treina e avalia diferentes arquiteturas de visão computacional para **classificação de imagens**.  
As métricas finais de cada modelo são registradas em `results/evaluation.csv` e os **pesos/modelos compilados** são salvos em `checkpoints/`.

## 🔧 Requisitos

- **Conda/Anaconda** instalado
- GPU NVIDIA (opcional, mas recomendado) com drivers/CUDA compatíveis
- Python e dependências serão instaladas via `environment.yml`

## 📦 Instalação

1. **Clonar o repositório** (ou baixar os arquivos do projeto).

2. **Criar a estrutura de dados**:

```
images/
  ├─ classe_1/
  │    ├─ img001.jpg
  │    └─ ...
  ├─ classe_2/
  │    ├─ img123.jpg
  │    └─ ...
  └─ ...
```

> Coloque **todas as classes** como subpastas dentro de `images/`, cada uma contendo suas imagens.

3. **Criar e ativar o ambiente Conda**:

```bash
conda env create -f environment.yml
conda activate <nome-do-ambiente>
```

## ▶️ Como Rodar

Na raiz do projeto:

```bash
python main.py
```

Ao finalizar, o script:

- **Gera as métricas** (Loss, Accuracy, Precision, Recall) por modelo em:  
  `results/evaluation.csv`
- **Salva pesos e modelos compilados** em:  
  `checkpoints/`

## 🧠 Modelos Disponíveis (8)

- CNN “pura”
- ConvNeXt
- DenseNet
- Inception
- NASNet
- ResNet50
- ResNet152
- Xception

> O pipeline treina/avalia cada um deles e **acrescenta** a linha correspondente no CSV final.

## 📄 Saídas

### 1) Métricas (CSV)

Arquivo: `results/evaluation.csv`  
Colunas:

- `Model` — nome do modelo
- `Loss`
- `Accuracy`
- `Precision`
- `Recall`

Cada execução **anexa** novas linhas ao CSV (não sobrescreve o cabeçalho se já existir).

### 2) Checkpoints

Pasta: `checkpoints/`

- Pesos (ex.: `*.h5`, `*.ckpt`)
- Modelo compilado/salvo (ex.: `saved_model/` ou `*.h5`, conforme implementação)

## 🗂️ Estrutura Recomendada do Projeto

```
.
├─ checkpoints/            # será criado automaticamente ao treinar
├─ datasets/               # são as classes imagens divididas em treinamento e teste
├─ images/                 # suas classes e imagens (obrigatório)
├─ models/                 # são todos os modelos de CNN.
├─ results/
│   └─ evaluation.csv      # criado/atualizado após rodar
├─ utils/                  # funções utilitárias.
├─ .gitignore
├─ config.yml              # configurações de hyperparametros
├─ environment.yml         # dependências do Conda
├─ main.py                 # ponto de entrada
└─ README.md
```

## ⚙️ Configuração (opcional)

Caso o projeto suporte, você pode controlar parâmetros (batch size, epochs, split, augmentations etc.) por variáveis em `main.py` ou arquivos de config.

## 🚀 Dicas de Desempenho

- Use **GPU** para acelerar o treino (verifique se o TensorFlow/PyTorch está reconhecendo a GPU no ambiente criado).
- Mantenha as imagens com tamanho razoável (ex.: 224x224) para equilibrar rapidez e acurácia.
- Se a base for desbalanceada, considere `class_weight` ou estratégias de oversampling.

## 🛠️ Solução de Problemas

- **`conda env create` falhou**: atualize o Conda (`conda update -n base -c defaults conda`) e tente novamente.
- **Sem GPU detectada**: verifique drivers/CUDA/cuDNN compatíveis com a versão instalada no `environment.yml`.
- **`images/` vazia ou nomes incorretos**: confere se as classes são subpastas diretas de `images/` e se há imagens dentro.
- **Permissão ao gravar em `results/`**: rode o script na raiz do projeto e verifique permissões da pasta.
