import streamlit as st
import pandas as pd
import joblib
import requests
import re
from datetime import timedelta

# ================= CONFIG ================= #

st.set_page_config(page_title="Video RAG + Text Assistant", layout="wide")

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
    td = timedelta(milliseconds=int(ms))
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ================= LOAD TRANSCRIPT (CACHE ONLY RAW LOAD) ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    return joblib.load("video2_embeddings_updated.joblib")

df = load_transcript()

# ================= PREPROCESS (DO NOT CACHE) ================= #

df = df.copy()
df["text_lower"] = df["text"].str.lower()
df["text_tokens"] = df["text_lower"].str.split().apply(set)

# ================= GENERATE RESPONSE ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are a senior Java instructor and software engineer.

Context:
{context}

Question:
{question}

Requirements:
- Clear explanation
- Java syntax & example
- Best practices / pitfalls
- Interview-ready answer
"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "web_access": False
    }

    try:
        response = requests.post(
            RAPIDAPI_URL,
            json=payload,
            headers=HEADERS,
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"LLM Error: {e}"

# ================= UI ================= #

st.title("🎥 Video Transcript RAG + Text Assistant")

query_input = st.text_input("Ask your question:")

if query_input.strip():
    query_tokens = set(query_input.lower().split())

    with st.spinner("Finding relevant transcript segments..."):
        df["score"] = df["text_tokens"].apply(
            lambda tokens: len(tokens & query_tokens)
        )

        top_snippets = (
            df[df["score"] > 0]
            .sort_values("score", ascending=False)
            .head(3)
        )

        if top_snippets.empty:
            context_text = ""
        else:
            context_blocks = []
            char_count = 0

            for _, row in top_snippets.iterrows():
                block = f"[{ms_to_hms(row.start)} - {ms_to_hms(row.end)}] {row.text}"
                if char_count + len(block) > 1500:
                    break
                context_blocks.append(block)
                char_count += len(block)

            context_text = "\n".join(context_blocks)

    st.subheader("🔎 Top Relevant Transcript Chunks")
    for _, row in top_snippets.iterrows():
        st.write(f"{ms_to_hms(row.start)} – {ms_to_hms(row.end)} | Score: {row.score}")
        st.write(row.text)
        st.markdown("---")

    answer = generate_llm_response(context_text, query_input)

    st.subheader("💬 Answer")
    st.markdown(answer)

    st.download_button("Download Answer", answer, "answer.txt")

else:
    st.info("Please enter a question to get started.")
