import anthropic
import base64

def translate_libras_with_claude(video_frames_base64: list) -> str:
    """
    Usa a API da Anthropic com visão para interpretar Libras
    (para sinais mais simples/estáticos)
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": [
                *[{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": frame
                    }
                } for frame in video_frames_base64[:5]],  # primeiros 5 frames
                {
                    "type": "text",
                    "text": "Esses frames são de um vídeo em Libras. Qual sinal/palavra está sendo feito? Responda apenas com a tradução em português."
                }
            ]
        }]
    )
    return response.content[0].text