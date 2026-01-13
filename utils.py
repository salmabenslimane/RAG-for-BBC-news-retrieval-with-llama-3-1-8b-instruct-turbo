import json
import numpy as np
import pandas as pd
from dateutil import parser
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from dotenv import load_dotenv
import ipywidgets as widgets
from IPython.display import display, Markdown

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
model_name = os.path.join(os.environ.get("MODEL_PATH", ""), "BAAI/bge-base-en-v1.5")
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
        return {"role": message["role"], "content": message["content"]}
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

def format_relevant_data(relevant_data):
    """
    Formats the top_k relevant documents into a string for RAG prompts.
    """
    formatted_documents = []
    for document in relevant_data:
        formatted_document = (
            f"Title: {document.get('title', '')}\n"
            f"Description: {document.get('description', '')}\n"
            f"Published at: {document.get('published_at', '')}\n"
            f"URL: {document.get('url', '')}\n"
        )
        formatted_documents.append(formatted_document)
    return "\n".join(formatted_documents)

NEWS_DATA = pd.read_csv("./news_data_dedup.csv").to_dict(orient="records")

def retrieve(query, top_k=5):
    query_embedding = model.encode(query)
    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), EMBEDDINGS)[0]
    top_indices = np.argsort(-similarity_scores)[:top_k]
    return top_indices

# ================================
# 🖥️ JUPYTER WIDGET UI
# ================================
def display_widget(llm_call_func):
    def on_button_click(b):
        output1.clear_output()
        output2.clear_output()
        status_output.clear_output()
        status_output.append_stdout("Generating...\n")

        query = query_input.value
        top_k = slider.value
        prompt = prompt_input.value.strip() if prompt_input.value.strip() else None

        response1 = llm_call_func(query, use_rag=True, top_k=top_k, prompt=prompt)
        response2 = llm_call_func(query, use_rag=False, top_k=top_k, prompt=prompt)

        with output1:
            display(Markdown(response1))
        with output2:
            display(Markdown(response2))
        status_output.clear_output()

    query_input = widgets.Text(
        description="Query:",
        placeholder="Type your query here",
        layout=widgets.Layout(width="100%")
    )

    prompt_input = widgets.Textarea(
        description="Augmented prompt layout:",
        placeholder=("Type your prompt layout here, don't forget to add {query} and {documents} "
                     "where you want them. Leave blank to use default."),
        layout=widgets.Layout(width="100%", height="100px"),
        style={"description_width": "initial"}
    )

    slider = widgets.IntSlider(
        value=5,
        min=1,
        max=20,
        step=1,
        description="Top K:",
        style={"description_width": "initial"}
    )

    output1 = widgets.Output(layout={"border": "1px solid #ccc", "width": "45%"})
    output2 = widgets.Output(layout={"border": "1px solid #ccc", "width": "45%"})
    status_output = widgets.Output()

    submit_button = widgets.Button(
        description="Get Responses",
        style={"button_color": "#f0f0f0", "font_color": "black"}
    )
    submit_button.on_click(on_button_click)

    label1 = widgets.Label(value="With RAG", layout={"width": "45%", "text_align": "center"})
    label2 = widgets.Label(value="Without RAG", layout={"width": "45%", "text_align": "center"})

    display(widgets.HTML("""
    <style>
        .custom-output {
            background-color: #f9f9f9;
            color: black;
            border-radius: 5px;
        }
        .widget-textarea, .widget-button {
            background-color: #f0f0f0 !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
        .widget-output {
            background-color: #f9f9f9 !important;
            color: black !important;
        }
        textarea {
            background-color: #fff !important;
            color: black !important;
            border: 1px solid #ccc !important;
        }
    </style>
    """))

    display(query_input, prompt_input, slider, submit_button, status_output)
    hbox_labels = widgets.HBox([label1, label2], layout={"justify_content": "space-between"})
    hbox_outputs = widgets.HBox([output1, output2], layout={"justify_content": "space-between"})

    def style_outputs(*outputs):
        for output in outputs:
            output.layout.margin = "5px"
            output.layout.height = "300px"
            output.layout.padding = "10px"
            output.layout.overflow = "auto"
            output.add_class("custom-output")

    style_outputs(output1, output2)
    display(hbox_labels)
    display(hbox_outputs)
