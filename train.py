"""
train_libras.py
---------------
Baixa o dataset LIBRAS Movement (UCI), treina um modelo LSTM e exporta:
  - libras_model.h5
  - labels.txt

Uso:
    python train_libras.py
    python train_libras.py --epochs 100 --output meu_modelo.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ──────────────────────────────────────────────
# 1. DOWNLOAD DO DATASET
# ──────────────────────────────────────────────

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data"

# 15 classes do dataset UCI LIBRAS
LABEL_NAMES = [
    "curved swing",
    "horizontal swing",
    "vertical swing",
    "anti-clockwise arc",
    "clockwise arc",
    "circle",
    "forward and back",
    "trefoil",
    "pinch open",
    "pinch close",
    "tap",
    "stop sign",
    "letter d",
    "f shape",
    "l shape",
]


def download_dataset(dest: Path) -> Path:
    csv_path = dest / "movement_libras.data"
    if csv_path.exists():
        print(f"Dataset já existe em {csv_path}, pulando download.")
        return csv_path

    print("Baixando dataset UCI LIBRAS...")
    dest.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()
    csv_path.write_bytes(response.content)
    print(f"Dataset salvo em {csv_path}")
    return csv_path


# ──────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ──────────────────────────────────────────────

def load_and_prepare(csv_path: Path):
    """
    O dataset UCI LIBRAS tem 90 colunas de coordenadas (x,y) de 45 pontos
    e a última coluna é a classe (1–15).
    Retorna X com shape (n_samples, 45, 2) e y one-hot.
    """
    df = pd.read_csv(csv_path, header=None)
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")

    X_raw = df.iloc[:, :-1].values.astype(np.float32)   # (360, 90)
    y_raw = df.iloc[:, -1].values.astype(int) - 1        # classes 1-15 → 0-14

    # Reshape para (n_samples, timesteps=45, features=2)
    X = X_raw.reshape(-1, 45, 2)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    y = to_categorical(y_enc, num_classes=num_classes)

    print(f"Classes encontradas: {num_classes}")
    return X, y, num_classes, le


# ──────────────────────────────────────────────
# 3. MODELO LSTM
# ──────────────────────────────────────────────

def build_model(timesteps: int, features: int, num_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────
# 4. TREINAMENTO
# ──────────────────────────────────────────────

def train(X, y, num_classes: int, epochs: int, output_path: Path):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Treino: {X_train.shape[0]} amostras | Validação: {X_val.shape[0]} amostras")

    timesteps, features = X.shape[1], X.shape[2]
    model = build_model(timesteps, features, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(output_path), save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nAcurácia final na validação: {acc * 100:.1f}%")
    return model, history


# ──────────────────────────────────────────────
# 5. EXPORTAR labels.txt
# ──────────────────────────────────────────────

def save_labels(le, labels_path: Path):
    """Salva os labels na ordem dos índices do LabelEncoder."""
    lines = []
    for idx in le.classes_:
        # le.classes_ contém os índices originais (0-14)
        lines.append(LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else str(idx))
    labels_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Labels salvos em {labels_path}")


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description="Treina modelo LSTM para Libras (UCI dataset)")
    parser.add_argument("--epochs",  type=int, default=200, help="Máximo de épocas (padrão: 200)")
    parser.add_argument("--output",  default="libras_model.h5", help="Arquivo .h5 de saída (padrão: libras_model.h5)")
    parser.add_argument("--labels",  default="labels.txt",      help="Arquivo de labels (padrão: labels.txt)")
    parser.add_argument("--data-dir",default="data",            help="Pasta para o dataset (padrão: data/)")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    labels_path = Path(args.labels)
    data_dir    = Path(args.data_dir)

    # Download
    csv_path = download_dataset(data_dir)

    # Pré-processamento
    X, y, num_classes, le = load_and_prepare(csv_path)

    # Treino
    model, _ = train(X, y, num_classes, epochs=args.epochs, output_path=output_path)

    # Labels
    save_labels(le, labels_path)

    print(f"\n✅ Pronto!")
    print(f"   Modelo : {output_path}")
    print(f"   Labels : {labels_path}")
    print(f"\nAgora rode:")
    print(f"   python translate_libras.py seu_video.mp4 --model {output_path} --labels {labels_path} --verbose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())