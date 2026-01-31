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
    td = timedelta(milliseconds=int(ms))
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ================= LOAD TRANSCRIPT ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    # Expected columns: start, end, text
    return joblib.load("video2_embeddings_updated.joblib")

@st.cache_data(show_spinner=False)
def preprocess_transcript(df):
    df = df.copy()
    df["text_lower"] = df["text"].astype(str).str.lower()
    # tuple() avoids Streamlit hashing error
    df["text_tokens"] = df["text_lower"].str.split().apply(lambda x: tuple(set(x)))
    return df

df = preprocess_transcript(load_transcript())

# ================= LLM ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are a senior Java instructor and software engineer.

If transcript context is useful, use it.
If not, answer purely from your own technical knowledge.

Context:
{context}

Question:
{question}

Answer requirements:
- Clear, technically accurate explanation
- Explain syntax and working
- Provide Java code example
- Mention best practices or pitfalls
- Professional, interview-level response
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
            timeout=40
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"LLM API Error: {e}"

# ================= UI ================= #

st.title("🎥 Video Transcript RAG + Text Assistant")

query_input = st.text_input(
    "Ask your question:",
    placeholder="Example: What are arithmetic operators in Java?"
)

if not query_input.strip():
    st.info("Please enter a question to get started.")
    st.stop()

query_tokens = set(query_input.lower().split())

with st.spinner("Searching transcript..."):
    df["score"] = df["text_tokens"].apply(
        lambda tokens: len(query_tokens.intersection(tokens))
    )

    top_snippets = (
        df[df["score"] > 0]
        .sort_values("score", ascending=False)
        .head(3)
    )

# ================= CONTEXT FILTER (CRITICAL FIX) ================= #

TOTAL_SCORE_THRESHOLD = 5

if top_snippets.empty or top_snippets["score"].sum() < TOTAL_SCORE_THRESHOLD:
    # Ignore transcript entirely if relevance is weak
    context_text = ""
else:
    context_blocks = []
    char_count = 0

    for _, row in top_snippets.iterrows():
        snippet = f"[{ms_to_hms(row.start)} - {ms_to_hms(row.end)}] {row.text}"
        if char_count + len(snippet) > 1500:
            break
        context_blocks.append(snippet)
        char_count += len(snippet)

    context_text = "\n".join(context_blocks)

# ================= DISPLAY TRANSCRIPT ================= #

st.subheader("🔎 Top Relevant Transcript Chunks")

if context_text:
    for _, row in top_snippets.iterrows():
        st.write(
            f"{ms_to_hms(row.start)} – {ms_to_hms(row.end)} | Score: {row.score}"
        )
        st.write(row.text)
        st.markdown("---")
else:
    st.write("Transcript context not relevant for this question.")

# ================= ANSWER ================= #

with st.spinner("Generating answer..."):
    answer = generate_llm_response(context_text, query_input)

st.subheader("💬 Answer")
st.markdown(answer)

st.download_button(
    "📥 Download Answer",
    answer,
    file_name="answer.txt"
)
