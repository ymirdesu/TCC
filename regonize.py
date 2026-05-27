"""
recognize.py
Recebe um arquivo de video com sinais LIBRAS e retorna uma string.

Uso:
    python recognize.py video.mp4
    python recognize.py video.mp4 --verbose
"""
import cv2
import numpy as np
import pickle
import sys
import os

MODELO_PATH = "modelo_libras.pkl"
FRAMES_POR_AMOSTRA = 30
THRESHOLD = 0.55        # confianca minima para aceitar um sinal
SOBREPOSICAO = 10       # frames de sobreposicao entre janelas (sliding window)


# ── carrega modelo ─────────────────────────────────────────────────
def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("ERRO: modelo_libras.pkl nao encontrado.")
        print("Rode primeiro: python train.py")
        sys.exit(1)
    with open(MODELO_PATH, "rb") as f:
        dados = pickle.load(f)
    return dados["modelo"], dados["labels"]


# ── mediapipe ──────────────────────────────────────────────────────
def iniciar_mediapipe():
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return hands


def extrair_landmarks(frame, hands):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
    coords -= np.tile(coords[:3], 21)
    return coords


def pad_ou_trim(seq, tamanho=30):
    seq = np.array(seq)
    if len(seq) >= tamanho:
        return seq[:tamanho]
    falta = tamanho - len(seq)
    return np.vstack([seq, np.repeat(seq[-1:], falta, axis=0)])


# ── reconhecimento principal ───────────────────────────────────────
def reconhecer_video(path_video, verbose=False):
    """
    Processa o video e retorna uma string com os sinais reconhecidos.
    """
    if not os.path.exists(path_video):
        print(f"ERRO: arquivo '{path_video}' nao encontrado.")
        sys.exit(1)

    modelo, labels = carregar_modelo()
    hands = iniciar_mediapipe()

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print(f"ERRO: nao foi possivel abrir '{path_video}'")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if verbose:
        print(f"Video: {path_video}")
        print(f"Frames: {total_frames} | FPS: {fps:.1f}")
        print(f"Duracao: {total_frames/fps:.1f}s\n")

    # extrai todos os landmarks do video
    todos_landmarks = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lm = extrair_landmarks(frame, hands)
        todos_landmarks.append(lm)  # None se mao nao detectada
        frame_idx += 1

        if verbose and frame_idx % 30 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"\r  Extraindo landmarks: {pct:.0f}%", end="", flush=True)

    cap.release()
    hands.close()

    if verbose:
        detectados = sum(1 for l in todos_landmarks if l is not None)
        print(f"\r  Landmarks extraidos: {detectados}/{len(todos_landmarks)} frames com mao\n")

    # sliding window: janelas de FRAMES_POR_AMOSTRA com sobreposicao
    sinais_detectados = []
    i = 0
    ultimo_sinal = None

    while i + FRAMES_POR_AMOSTRA <= len(todos_landmarks):
        janela = todos_landmarks[i:i + FRAMES_POR_AMOSTRA]

        # filtra frames com mao detectada
        frames_validos = [f for f in janela if f is not None]

        if len(frames_validos) >= FRAMES_POR_AMOSTRA // 2:
            # completa com pad se necessario
            seq = pad_ou_trim(frames_validos, FRAMES_POR_AMOSTRA)
            features = seq.flatten().reshape(1, -1)

            probs = modelo.predict_proba(features)[0]
            conf = probs.max()
            pred_idx = probs.argmax()
            sinal = labels[pred_idx]

            if conf >= THRESHOLD:
                # evita repetir o mesmo sinal consecutivamente
                if sinal != ultimo_sinal:
                    sinais_detectados.append((sinal, conf, i))
                    ultimo_sinal = sinal
                    if verbose:
                        tempo = i / fps if fps > 0 else i
                        print(f"  [{tempo:.1f}s] {sinal} ({conf:.0%})")
                i += FRAMES_POR_AMOSTRA  # pula janela completa apos deteccao
            else:
                i += SOBREPOSICAO  # avanca com sobreposicao
        else:
            i += SOBREPOSICAO

    # monta string final
    frase = " ".join([s[0] for s in sinais_detectados])
    return frase


# ── entrada principal ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python recognize.py <video.mp4> [--verbose]")
        print("Exemplo: python recognize.py meu_sinal.mp4 --verbose")
        sys.exit(1)

    path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Processando: {path}")
    resultado = reconhecer_video(path, verbose=verbose)

    print("\n" + "=" * 45)
    print("RESULTADO:")
    print(f"  {resultado if resultado else '(nenhum sinal reconhecido)'}")
    print("=" * 45)