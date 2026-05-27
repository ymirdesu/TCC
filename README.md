## version2 — Reconhecimento de sinais LIBRAS (MINDS-Libras)

Este projeto faz um **pipeline simples de reconhecimento de sinais**:

- **`setup.py`**: baixa o dataset **MINDS-Libras**, extrai e processa os vídeos, e salva **landmarks de mão (MediaPipe)** em `dados/<sinal>/*.npy`
- **`train.py`**: treina um classificador (RandomForest) com os `.npy` e salva `modelo_libras.pkl`
- **`regonize.py`**: reconhece sinais em um vídeo `.mp4` usando `modelo_libras.pkl` e imprime a frase detectada  
  (observação: o nome do arquivo está como `regonize.py`, mas o “uso” dentro do script fala `recognize.py`)

## Requisitos

- Python **3.9+**
- Windows (instruções abaixo em PowerShell)

Dependências principais (instalação via `pip`):
- `mediapipe`, `opencv-python`, `numpy`, `torch`
- `scikit-learn` (necessário para `train.py`)

## Como rodar (passo a passo)

### 1) Criar e ativar ambiente virtual (recomendado)

No PowerShell, dentro da pasta do projeto:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Instalar dependências

```bash
pip install mediapipe opencv-python numpy torch scikit-learn
```

Se você tiver problemas com `torch` (CUDA/versão), instale conforme a página oficial do PyTorch (CPU ou CUDA).

### 3) Baixar dataset e gerar `dados/` (landmarks)

Execute **uma vez**:

```bash
python setup.py
```

Isso vai:
- baixar `minds_libras.zip`
- extrair para `MINDS-Libras-master/`
- criar `dados/` com amostras por sinal
- salvar `labels.txt` com a lista de sinais encontrados

### 4) Treinar o modelo

```bash
python train.py
```

Ao final, será criado:
- `modelo_libras.pkl`

### 5) Reconhecer sinais em um vídeo

Passe o caminho do vídeo (ex.: `.mp4`):

```bash
python regonize.py caminho\para\video.mp4
```

Modo verboso (mostra progresso e detecções com timestamp):

```bash
python regonize.py caminho\para\video.mp4 --verbose
```

## Estrutura gerada/arquivos importantes

- **`dados/`**: amostras por sinal (arquivos `.npy` com sequência de \(30\) frames)
- **`labels.txt`**: lista de sinais (classes)
- **`modelo_libras.pkl`**: modelo treinado + labels

## Dicas rápidas / problemas comuns

- **`modelo_libras.pkl nao encontrado`**: rode `python train.py` primeiro.
- **`dados/ nao encontrada`**: rode `python setup.py` primeiro.
- **Pouca detecção de mão**: iluminação ruim, mão fora do frame ou movimento muito rápido pode fazer o MediaPipe falhar.
