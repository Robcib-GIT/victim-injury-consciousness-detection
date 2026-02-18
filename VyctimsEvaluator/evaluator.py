import requests # type: ignore
import json
import base64

URL = "http://127.0.0.1:1234/v1/chat/completions"


def ask_url(crop_bytes, question, choices):
    image_b64 = base64.b64encode(crop_bytes).decode("utf-8")

    prompt = f"""
Answer ONLY with one of these options:
{choices}

Question:
{question}
"""

    payload = {
        "model": "qwen/qwen3-vl-30b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                ],
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer no-key-needed",
    }

    r = requests.post(URL, headers=headers, json=payload)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


def ask_one_question(*, crop_bytes: bytes, question: str, choices: list):
    answer = ask_url(
        crop_bytes=crop_bytes,
        question=question,
        choices=choices
    )
    return answer
