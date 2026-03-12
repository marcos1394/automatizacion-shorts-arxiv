#!/usr/bin/env python3
# arxiv_to_x_bot.py
# Versión corregida para usar Qwen 2.5 (que ya tienes local)

import os
import arxiv
import tweepy
from dotenv import load_dotenv
import time
from datetime import datetime
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ==================== CONFIGURACIÓN ====================
load_dotenv()

# IMPORTANTE: Usamos el modelo que SÍ funciona (Qwen 2.5, no 3.5)
MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
print(f"Cargando modelo {MODEL_NAME} (usando caché local)...")
model, tokenizer = load(MODEL_NAME)
print("✅ Modelo cargado exitosamente.")

# ==================== FUNCIONES ====================
def get_x_client():
    """Cliente de X (Twitter)"""
    api_key = os.getenv("X_API_KEY")
    api_secret = os.getenv("X_API_SECRET")
    access_token = os.getenv("X_ACCESS_TOKEN")
    access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
    return tweepy.Client(consumer_key=api_key, consumer_secret=api_secret,
                         access_token=access_token, access_token_secret=access_token_secret)

def fetch_latest_paper():
    """Último paper de cs.AI en arXiv"""
    search = arxiv.Search(query="cat:cs.AI", max_results=1, sort_by=arxiv.SortCriterion.SubmittedDate)
    return next(search.results())

def generate_thread(paper):
    """Genera un hilo usando Qwen 2.5 con el sampler correcto."""
    prompt = f"""Actúa como divulgador científico en X. Crea un hilo de 3-5 tweets sobre:
    Título: {paper.title}
    Resumen: {paper.summary}

    Requisitos:
    - Hook inicial atractivo
    - Problema que resuelve
    - 3 hallazgos clave
    - Pregunta final
    - Enlace: {paper.entry_id}

    Separa cada tweet con '---'."""

    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ---> ESTA ES LA PARTE CLAVE <---
    # 1. Creamos un sampler con la temperatura que queremos (0.7 es un buen valor)
    sampler = make_sampler(temp=0.7)

    # 2. Llamamos a 'generate' y le pasamos el sampler con el argumento 'sampler'
    response = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=1000,
        sampler=sampler  # Aquí va el sampler, no 'temperature'
    )
    return response


def parse_thread(text):
    """Convierte texto a lista de tweets"""
    tweets = [t.strip() for t in text.split('---') if t.strip()]
    return [t[:277] + "..." if len(t) > 280 else t for t in tweets]

def publish_thread(client, tweets):
    """Publica hilo en X"""
    prev_id = None
    for i, tweet in enumerate(tweets):
        if i == 0:
            resp = client.create_tweet(text=tweet)
        else:
            resp = client.create_tweet(text=tweet, in_reply_to_tweet_id=prev_id)
        prev_id = resp.data['id']
        time.sleep(2)
    return True

# ==================== MAIN ====================
def main():
    paper = fetch_latest_paper()
    thread_text = generate_thread(paper)
    tweets = parse_thread(thread_text)
    client = get_x_client()
    publish_thread(client, tweets)
    print("✅ Hilo publicado!")

if __name__ == "__main__":
    main()