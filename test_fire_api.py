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

# ================= LOAD TRANSCRIPT ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    # DataFrame expected with columns: start, end, text
    return joblib.load("video2_embeddings_updated.joblib")

@st.cache_data(show_spinner=False)
def preprocess_transcript_texts(df):
    df = df.copy()
    df["text_lower"] = df["text"].str.lower()
    # Store tokens as sorted tuple for caching compatibility
    df["text_tokens"] = df["text_lower"].str.split().apply(lambda tokens: tuple(sorted(tokens)))
    return df

df = load_transcript()
df = preprocess_transcript_texts(df)

# ================= GENERATE RESPONSE ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are a senior Java instructor and software engineer.

Use the transcript context ONLY if it is directly relevant.
Do NOT quote timestamps unless they add real value.

Context (may be empty):
{context}

Question:
{question}

Answer requirements:
1. Give a clear, concise, and technically accurate explanation.
2. Explain syntax, working, and common use-cases.
3. Include at least one Java code example.
4. Mention pitfalls or best practices if applicable.
5. If the transcript lacks useful info, answer from general knowledge without saying so explicitly.

Keep the answer professional and advanced, suitable for interview or academic use.
"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "web_access": False
    }
    try:
        response = requests.post(RAPIDAPI_URL, json=payload, headers=HEADERS, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"Error contacting LLM API: {str(e)}"

# ================= UI ================= #

st.title("🎥 Video Transcript RAG + Text Assistant")

query_input = st.text_input("Ask your question here:")

if query_input.strip():
    query = query_input.lower().split()
    query_tokens = set(query)

    with st.spinner("Finding relevant transcript segments..."):
        df["score"] = df["text_tokens"].apply(lambda tokens: len(query_tokens.intersection(tokens)))
        top_snippets = df[df["score"] > 0].sort_values("score", ascending=False).head(3)

        if top_snippets.empty:
            context_text = "No relevant transcript found."
        else:
            context_blocks = []
            char_count = 0
            for _, row in top_snippets.iterrows():
                start = ms_to_hms(row.start)
                end = ms_to_hms(row.end)
                snippet_text = f"[{start} - {end}] {row.text}"
                if char_count + len(snippet_text) > 1500:
                    break
                context_blocks.append(snippet_text)
                char_count += len(snippet_text)
            context_text = "\n".join(context_blocks)

    st.subheader("🔎 Top Relevant Transcript Chunks")
    if top_snippets.empty:
        st.write("No relevant transcript chunks found.")
    else:
        for _, row in top_snippets.iterrows():
            start = ms_to_hms(row.start)
            end = ms_to_hms(row.end)
            st.write(f"{start} – {end} | Score: {row.score}")
            st.write(row.text)
            st.markdown("---")

    answer = generate_llm_response(context_text, query_input)

    st.subheader("💬 Answer")
    st.markdown(answer)

    st.download_button("Download Answer", answer, file_name="answer.txt")

else:
    st.info("Please enter a question to get started.")
