import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import tempfile
import os

from datetime import timedelta
from faster_whisper import WhisperModel

# ================= CONFIG ================= #

st.set_page_config(page_title="Video RAG + Voice Assistant", layout="wide")

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
    if h:
        return f"{h:02}:{m:02}:{s:02}"
    else:
        return f"{m:02}:{s:02}"

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ================= WHISPER ================= #

@st.cache_resource(show_spinner=False)
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="float32")

whisper_model = load_whisper()

def transcribe_audio(audio_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.getvalue())
        path = f.name
    try:
        segments, _ = whisper_model.transcribe(
            path,
            beam_size=1,
            language="en",
            temperature=0.0,
            vad_filter=True
        )
        text = " ".join(seg.text for seg in segments)
        return normalize_text(text)
    finally:
        os.remove(path)

# ================= LOAD TRANSCRIPT ================= #

@st.cache_data(show_spinner=False)
def load_transcript():
    # Expect df with columns: start, end, text
    return joblib.load("video2_embeddings_updated.joblib")

df = load_transcript()

# Preprocess transcript text once for faster matching
@st.cache_data(show_spinner=False)
def preprocess_transcript_texts(df):
    # Create new column with token sets for faster keyword matching
    df = df.copy()
    df["text_lower"] = df["text"].str.lower()
    df["text_tokens"] = df["text_lower"].str.split().apply(set)
    return df

df = preprocess_transcript_texts(df)

# ================= GENERATE RESPONSE ================= #

def generate_llm_response(context: str, question: str) -> str:
    prompt = f"""
You are an expert assistant answering questions about a video transcript.

Here are transcript excerpts for context (if any):
{context}

Question:
{question}

Instructions:
- If the transcript excerpts do NOT contain relevant information to answer the question, clearly state that the transcript does not cover this topic.
- Then provide a helpful answer based on your general knowledge.
- If relevant, mention timestamps from the transcript excerpts.
- Keep the answer clear and concise.
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

st.title("🎥 Video Transcript RAG + Voice Assistant")

st.subheader("Ask a question by voice or text")

col1, col2, col3 = st.columns([6, 2, 1])

with col1:
    query_input = st.text_input("Your question:", value=st.session_state.get("query", ""))

with col2:
    audio = st.audio_input("Or speak your question")

with col3:
    if st.button("Clear"):
        st.session_state.query = ""
        st.session_state.run_rag = False
        st.experimental_rerun()

# Handle voice input

if audio and not st.session_state.get("run_rag", False):
    with st.spinner("Transcribing audio..."):
        text = transcribe_audio(audio)
        if text:
            st.session_state.query = text
            st.session_state.run_rag = True
            st.success(f"Recognized: {text}")
            st.experimental_rerun()

# Handle text input

if query_input.strip() and query_input.strip() != st.session_state.get("query", ""):
    st.session_state.query = query_input.strip()
    st.session_state.run_rag = True

# RAG + LLM answering

if st.session_state.get("run_rag", False) and st.session_state.get("query", ""):
    query = st.session_state.query.lower().split()
    query_tokens = set(query)

    with st.spinner("Finding relevant transcript segments..."):
        # Compute relevance by intersection count with preprocessed tokens
        df["score"] = df["text_tokens"].apply(lambda tokens: len(query_tokens.intersection(tokens)))

        top_snippets = df[df["score"] > 0].sort_values("score", ascending=False).head(3)

        if top_snippets.empty:
            context_text = "No relevant transcript found."
        else:
            # Limit context length to ~1500 characters to avoid prompt overload
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

    # Display relevant transcript snippets with timestamps and score
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

    answer = generate_llm_response(context_text, st.session_state.query)

    st.subheader("💬 Answer")
    st.markdown(answer)

    st.download_button("Download Answer", answer, file_name="answer.txt")

    st.session_state.run_rag = False

else:
    st.info("Ask a question using text or voice.")
