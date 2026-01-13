import json
import numpy as np
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os

from dotenv import load_dotenv  # ✅ NEW

# ================================
# 🔐 LOAD ENV VARIABLES
# ================================

load_dotenv()  # reads .env automatically

OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct-turbo"

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_KEY not found in .env file")

# ================================
# 📦 EMBEDDING MODEL
# ================================

model_name = os.path.join(
    os.environ.get("MODEL_PATH", ""),
    "BAAI/bge-base-en-v1.5"
)

model = SentenceTransformer(model_name)

EMBEDDINGS = joblib.load("embeddings.joblib")

# ================================
# 🛠️ HELPERS
# ================================

def pprint(obj):
    print(json.dumps(obj, indent=2))


def format_date(date_string):
    date_object = parser.parse(date_string)
    return date_object.strftime("%Y-%m-%d")


def read_dataframe(path):
    df = pd.read_csv(path)
    df["published_at"] = df["published_at"].apply(format_date)
    df["updated_at"] = df["updated_at"].apply(format_date)
    return df.to_dict(orient="records")

# ================================
# 🤖 LLM CALL (OPENROUTER)
# ================================

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    top_p: float | None = None,
    temperature: float | None = None,
    max_tokens: int = 500,
    model: str = LLM_MODEL,
):
    payload = {
        "model": model,
        "messages": [{"role": role, "content": prompt}],
        "max_tokens": max_tokens,
    }

    if top_p is not None:
        payload["top_p"] = top_p
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG News App",
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if not response.ok:
        raise Exception(f"OpenRouter error: {response.text}")

    data = response.json()

    try:
        message = data["choices"][0]["message"]
        return {
            "role": message["role"],
            "content": message["content"],
        }
    except KeyError as e:
        raise Exception(f"Malformed response: {e}\n{data}")

# ================================
# 📚 RAG UTILITIES
# ================================

def concatenate_fields(dataset, fields):
    concatenated_data = []

    for data in dataset:
        text = ""
        for field in fields:
            value = data.get(field, "")
            if value:
                text += f"{value} "
        concatenated_data.append(text.strip()[:493])

    return concatenated_data


NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient="records")


def retrieve(query, top_k=5):
    query_embedding = model.encode(query)
    similarity_scores = cosine_similarity(
        query_embedding.reshape(1, -1), EMBEDDINGS
    )[0]

    top_indices = np.argsort(-similarity_scores)[:top_k]
    return top_indices
