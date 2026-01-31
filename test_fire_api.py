import streamlit as st
import pandas as pd
import joblib
import requests
import re
from datetime import timedelta

# ================= CONFIG ================= #

st.set_page_config(
    page_title="Video RAG + Text Assistant",
    layout="wide"
)

RAPIDAPI_URL = "https://open-ai21.p.rapidapi.com/conversationllama"
RAPIDAPI_KEY = "dcabac9b79msh4bbb16cbc29a17ap1d3c84jsncb0dbb461d84"
RAPIDAPI_HOST = "open-ai21.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
    "Content-Type": "application/json"
}

# ================= UTILITIES ================= #

def ms_to_hms(ms):
    try:
        td = timedelta(milliseconds=int(ms))
        total_seconds = int(td.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"
    except Exception:
        return "00:00"

def normalize_text(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()

# ================= LOAD TRANSCRIPT ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    return joblib.load("video2_embeddings_updated.joblib")

@st.cache_data(show_spinner=False)
def preprocess_transcript(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "text" not in df.columns:
        df["text"] = ""

    df["clean_text"] = df["text"].apply(normalize_text)
    df["text_tokens"] = df["clean_text"].apply(lambda x: set(x.split()))

    if "start" not in df.columns:
        df["start"] = 0
    if "end" not in df.columns:
        df["end"] = 0

    return df

df = preprocess_transcript(load_transcript())

# ================= LLM ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are a senior Java instructor and software engineer.

Use the transcript context ONLY if it is relevant.
If context is empty or weak, answer purely from your own knowledge.

Transcript Context:
{context if context else "N/A"}

Question:
{question}

Answer Requirements:
- Clear explanation
- Java syntax
- Example code
- Best practices
- Interview-ready response
"""

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "web_access": False
    }

    try:
        response = requests.post(
            RAPIDAPI_URL,
            headers=HEADERS,
            json=payload,
            timeout=45
        )
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            if "response" in data and data["response"]:
                return data["response"].strip()
            if "result" in data and data["result"]:
                return data["result"].strip()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()

        return "⚠️ LLM returned an empty response."

    except Exception as e:
        return f"❌ LLM API Error: {e}"

# ================= UI ================= #

st.title("🎥 Video Transcript RAG + Text Assistant")

query_input = st.text_input(
    "Ask your question:",
    placeholder="Example: What are arithmetic operators in Java?"
)

if not query_input.strip():
    st.info("Please enter a question to get started.")
    st.stop()

query_tokens = set(normalize_text(query_input).split())

# ================= SEARCH ================= #

with st.spinner("🔎 Searching transcript..."):
    df["score"] = df["text_tokens"].map(
        lambda tokens: len(tokens & query_tokens)
    )

    top_snippets = (
        df[df["score"] > 0]
        .sort_values("score", ascending=False)
        .head(1)   # ✅ ONLY TOP CHUNK
    )

# ================= CONTEXT FILTER ================= #

if top_snippets.empty:
    context_text = ""
else:
    row = top_snippets.iloc[0]
    context_text = f"[{ms_to_hms(row.start)} – {ms_to_hms(row.end)}] {row.text}"

# ================= DISPLAY TRANSCRIPT ================= #

st.subheader("🔎 Top Relevant Transcript Chunk")

if not top_snippets.empty:
    row = top_snippets.iloc[0]
    st.write(
        f"{ms_to_hms(row.start)} – {ms_to_hms(row.end)} | Score: {row.score}"
    )
    st.write(row.text)
else:
    st.write("No relevant transcript sections found.")

# ================= ANSWER ================= #

st.subheader("💬 Answer")

with st.spinner("🤖 Generating answer..."):
    answer = generate_llm_response(context_text, query_input)

st.markdown(answer)

st.download_button(
    "📥 Download Answer",
    answer,
    file_name="answer.txt",
    use_container_width=True
)
