"""
train.py
Treina o classificador de sinais LIBRAS com os dados extraidos pelo setup.py.
"""
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PASTA_DADOS = "dados"
MODELO_SAIDA = "modelo_libras.pkl"
FRAMES_POR_AMOSTRA = 30


def carregar_dados():
    X, y = [], []
    labels = sorted(os.listdir(PASTA_DADOS))
    labels = [l for l in labels if os.path.isdir(os.path.join(PASTA_DADOS, l))]

    print("Carregando amostras:")
    for idx, sinal in enumerate(labels):
        pasta = os.path.join(PASTA_DADOS, sinal)
        arquivos = [f for f in os.listdir(pasta) if f.endswith(".npy")]
        for arq in arquivos:
            seq = np.load(os.path.join(pasta, arq))  # (30, 63)
            X.append(seq.flatten())                   # (1890,)
            y.append(idx)
        print(f"  {sinal}: {len(arquivos)} amostras")

    return np.array(X), np.array(y), labels


def treinar():
    if not os.path.exists(PASTA_DADOS):
        print("ERRO: pasta 'dados/' nao encontrada.")
        print("Rode primeiro: python setup.py")
        input("Enter para fechar...")
        return

    print("=" * 45)
    print("  Treinamento - Reconhecimento LIBRAS")
    print("=" * 45)

    X, y, labels = carregar_dados()
    print(f"\nTotal: {len(X)} amostras | {len(labels)} sinais")

    if len(X) < 10:
        print("ERRO: amostras insuficientes para treinar.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")
    print("\nTreinando modelo...")

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAcuracia: {acc:.1%}")
    print("\nRelatorio por sinal:")
    print(classification_report(y_test, y_pred, target_names=labels))

    # salva modelo + labels
    with open(MODELO_SAIDA, "wb") as f:
        pickle.dump({"modelo": modelo, "labels": labels}, f)

    print(f"Modelo salvo em: {MODELO_SAIDA}")
    print("\nAgora rode: python recognize.py <caminho_do_video.mp4>")


if __name__ == "__main__":
    treinar()
