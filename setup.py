"""
setup.py
Baixa o dataset MINDS-Libras e extrai landmarks com MediaPipe.
Execute uma vez antes de treinar.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import zipfile

PASTA_DADOS = "dados"
PASTA_VIDEOS = "videos_libras"

# MINDS-Libras: 20 sinais, disponivel publicamente
DATASET_URL = "https://github.com/rafaelpadilla/MINDS-Libras/archive/refs/heads/master.zip"
DATASET_ZIP = "minds_libras.zip"

FRAMES_POR_AMOSTRA = 30

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)


def extrair_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
    coords -= np.tile(coords[:3], 21)
    return coords


def pad_ou_trim(seq, tamanho=30):
    """Garante que a sequencia tenha exatamente `tamanho` frames."""
    seq = np.array(seq)
    if len(seq) >= tamanho:
        return seq[:tamanho]
    # repete o ultimo frame para completar
    falta = tamanho - len(seq)
    ultimo = seq[-1:]
    return np.vstack([seq, np.repeat(ultimo, falta, axis=0)])


def processar_video(path_video):
    """Extrai sequencia de landmarks de um video."""
    cap = cv2.VideoCapture(path_video)
    seq = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lm = extrair_landmarks(frame)
        if lm is not None:
            seq.append(lm)
    cap.release()
    if len(seq) < 5:
        return None
    return pad_ou_trim(seq, FRAMES_POR_AMOSTRA)


def baixar_dataset():
    if os.path.exists(DATASET_ZIP):
        print("Arquivo zip ja existe, pulando download.")
        return
    print(f"Baixando MINDS-Libras...")
    print("(pode demorar alguns minutos)")

    def progresso(count, block, total):
        pct = min(count * block / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP, progresso)
    print("\nDownload concluido!")


def extrair_zip():
    if os.path.exists("MINDS-Libras-master"):
        print("Dataset ja extraido.")
        return
    print("Extraindo zip...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as z:
        z.extractall(".")
    print("Extraido!")


def processar_dataset():
    """Percorre os videos e salva landmarks em dados/<sinal>/<idx>.npy"""
    os.makedirs(PASTA_DADOS, exist_ok=True)

    # tenta encontrar a pasta de videos dentro do repositorio extraido
    base = "MINDS-Libras-master"
    if not os.path.exists(base):
        print(f"ERRO: pasta '{base}' nao encontrada.")
        print("Verifique se o download foi concluido.")
        return

    # lista sinais (subpastas)
    sinais_encontrados = []
    for root, dirs, files in os.walk(base):
        for d in dirs:
            pasta = os.path.join(root, d)
            videos = [f for f in os.listdir(pasta)
                      if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if videos:
                sinais_encontrados.append((d, pasta, videos))

    if not sinais_encontrados:
        print("\nNenhum video encontrado no dataset.")
        print("Estrutura encontrada:")
        for root, dirs, files in os.walk(base):
            nivel = root.replace(base, '').count(os.sep)
            print('  ' * nivel + os.path.basename(root) + '/')
            for f in files[:3]:
                print('  ' * (nivel+1) + f)
        return

    print(f"\nEncontrados {len(sinais_encontrados)} sinais.")
    labels = []

    for sinal, pasta, videos in sinais_encontrados:
        pasta_saida = os.path.join(PASTA_DADOS, sinal)
        os.makedirs(pasta_saida, exist_ok=True)
        labels.append(sinal)
        ok = 0
        for i, video in enumerate(videos):
            path = os.path.join(pasta, video)
            seq = processar_video(path)
            if seq is not None:
                np.save(os.path.join(pasta_saida, f"{i}.npy"), seq)
                ok += 1
        print(f"  {sinal}: {ok}/{len(videos)} videos processados")

    # salva lista de labels
    with open("labels.txt", "w") as f:
        f.write("\n".join(labels))

    print(f"\nPronto! {len(labels)} sinais salvos em '{PASTA_DADOS}/'")
    print("Agora rode: python train.py")


if __name__ == "__main__":
    baixar_dataset()
    extrair_zip()
    processar_dataset()