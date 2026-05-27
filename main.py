import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model  # ✅ keras standalone (TF 2.16+)
import argparse
import sys
import traceback
from pathlib import Path

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

def extract_keypoints(frame, holistic):
    """Extrai landmarks de mãos, rosto e corpo do frame"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Landmarks das mãos (21 pontos x 3 coords cada)
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([lh, rh])  # 126 features

def translate_video(video_path: str, model, label_map: dict, sequence_length=30, verbose: bool = False) -> str:
    """
    Recebe um vídeo em Libras e retorna a tradução em String
    """
    cap = cv2.VideoCapture(video_path)
    sequence = []
    sentence = []
    threshold = 0.85
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_i = 0

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = extract_keypoints(frame, holistic)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]  # janela deslizante

            if len(sequence) == sequence_length:
                input_data = np.expand_dims(sequence, axis=0)
                prediction = model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class]

                if confidence > threshold:
                    word = label_map.get(predicted_class, str(predicted_class))
                    if not sentence or sentence[-1] != word:
                        sentence.append(word)

            frame_i += 1
            if verbose and frame_i % 30 == 0:
                if total_frames > 0:
                    pct = (frame_i / total_frames) * 100
                    print(f"\rProcessando frames: {frame_i}/{total_frames} ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\rProcessando frames: {frame_i}", end="", flush=True)
    finally:
        holistic.close()

    cap.release()
    if verbose:
        print()
    return " ".join(sentence)  # String traduzida


def _load_labels_txt(labels_path: str) -> dict:
    """
    Lê um arquivo com 1 label por linha e monta {indice: label}.
    Ex.: labels.txt gerado pelo setup.py (ou um arquivo equivalente).
    """
    p = Path(labels_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {labels_path}")
    labels = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Arquivo de labels está vazio: {labels_path}")
    return {i: lbl for i, lbl in enumerate(labels)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tradução de vídeo em Libras usando um modelo Keras (.h5) + labels.txt"
    )
    parser.add_argument("video", help="Caminho do vídeo (ex.: video.mp4)")
    parser.add_argument(
        "--model",
        default="libras_model.h5",
        help="Caminho do modelo Keras (.h5). Padrão: libras_model.h5",
    )
    parser.add_argument(
        "--labels",
        default="labels.txt",
        help="Arquivo com 1 label por linha. Padrão: labels.txt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do arquivo .txt de saída. Padrão: mesmo nome do vídeo com extensão .txt",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Tamanho da janela temporal (frames). Padrão: 30",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra progresso enquanto processa o vídeo",
    )
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERRO: vídeo não encontrado: {video_path}", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERRO: modelo não encontrado: {model_path}", file=sys.stderr)
        print("Dica: passe --model caminho\\para\\seu_modelo.h5", file=sys.stderr)
        return 2

    label_map = _load_labels_txt(args.labels)
    model = load_model(str(model_path))

    resultado = translate_video(
        str(video_path),
        model,
        label_map,
        sequence_length=args.sequence_length,
        verbose=args.verbose
    )

    texto_final = resultado if resultado else "(nenhum sinal reconhecido)"

    # Salva no .txt com o mesmo nome do vídeo
    output_path = Path(args.output) if args.output else video_path.with_suffix(".txt")
    output_path.write_text(texto_final, encoding="utf-8")

    print(texto_final)
    print(f"\nTraducão salva em: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)