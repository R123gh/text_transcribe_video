import streamlit as st
import pandas as pd
import joblib
import requests
import re
import time
from datetime import timedelta

# ================= CONFIG ================= #

st.set_page_config(
    page_title="Video RAG + Text Assistant",
    layout="wide"
)

# 🔐 USE STREAMLIT SECRETS (IMPORTANT FOR CLOUD)
RAPIDAPI_URL = "https://open-ai21.p.rapidapi.com/conversationllama"
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "open-ai21.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
    "Content-Type": "application/json"
}

# ================= UTILITIES ================= #

def ms_to_hms(ms):
    td = timedelta(milliseconds=int(ms))
    s = int(td.total_seconds())
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()

# ================= LOAD TRANSCRIPT ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    df = joblib.load("video2_embeddings_updated.joblib")
    return df[["start", "end", "text"]]

@st.cache_data(show_spinner=False)
def preprocess_transcript(df):
    df = df.copy()
    df["text_clean"] = df["text"].apply(normalize_text)

    # ✅ tuple() avoids Streamlit hashing error
    df["tokens"] = df["text_clean"].str.split().apply(lambda x: tuple(set(x)))
    return df

df = preprocess_transcript(load_transcript())

# ================= LLM ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are a senior Java instructor.

Answer clearly and professionally.

Context:
{context[:1000] if context else "N/A"}

Question:
{question}

Requirements:
- Explain concept
- Java syntax
- Java code example
- Best practices
"""

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "web_access": False
    }

    for attempt in range(2):  # 🔁 retry once
        try:
            timeout = 30 if attempt == 0 else 60

            response = requests.post(
                RAPIDAPI_URL,
                headers=HEADERS,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            if "response" in data and data["response"].strip():
                return data["response"].strip()

            return "⚠️ LLM returned an empty response."

        except requests.exceptions.Timeout:
            if attempt == 1:
                return (
                    "⚠️ The AI service is currently slow.\n\n"
                    "Please try again after a few seconds."
                )
            time.sleep(1)

        except Exception as e:
            return f"❌ LLM API Error: {e}"

# ================= UI ================= #

st.title("🎥 Video Transcript RAG + Text Assistant")

query = st.text_input(
    "Ask your question:",
    placeholder="Example: What are arithmetic operators in Java?"
)

if not query.strip():
    st.info("Please enter a question to continue.")
    st.stop()

query_tokens = set(normalize_text(query).split())

# ================= SEARCH TRANSCRIPT ================= #

with st.spinner("Searching transcript..."):
    df["score"] = df["tokens"].apply(
        lambda t: len(query_tokens.intersection(t))
    )

    top_snippets = (
        df[df["score"] > 0]
        .sort_values("score", ascending=False)
        .head(3)
    )

# ================= CONTEXT FILTER ================= #

MIN_TOTAL_SCORE = 5

if top_snippets.empty or top_snippets["score"].sum() < MIN_TOTAL_SCORE:
    context_text = ""
else:
    context_blocks = []
    chars = 0

    for _, row in top_snippets.iterrows():
        block = f"[{ms_to_hms(row.start)} - {ms_to_hms(row.end)}] {row.text}"
        if chars + len(block) > 1200:
            break
        context_blocks.append(block)
        chars += len(block)

    context_text = "\n".join(context_blocks)

# ================= DISPLAY TRANSCRIPT ================= #

st.subheader("🔎 Relevant Transcript Segments")

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
    answer = generate_llm_response(context_text, query)

st.subheader("💬 Answer")

if answer.strip():
    st.markdown(answer)
else:
    st.warning("No response generated. Please try again.")

st.download_button(
    "📥 Download Answer",
    answer,
    file_name="answer.txt"
)
