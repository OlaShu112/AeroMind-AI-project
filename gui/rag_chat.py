import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="AeroMind AI",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 AeroMind AI")
st.caption("Drone Intelligence | RAG Knowledge Assistant")

st.info(
    "Answers are advisory. Flight control and safety "
    "validation remain independent of this interface."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Knowledge sources"):
                st.json(message["sources"])

question = st.chat_input(
    "Ask AeroMind AI a question..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and thinking..."):
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "ai.model",
                        "ask",
                        question,
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=360,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        result.stderr.strip()
                        or "AeroMind AI request failed."
                    )

                response = json.loads(result.stdout)

                answer = response.get(
                    "answer",
                    "No answer was generated."
                )

                sources = response.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Knowledge sources"):
                        st.json(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except subprocess.TimeoutExpired:
                st.error(
                    "The AI request timed out. "
                    "Check Ollama and try again."
                )

            except Exception as exc:
                st.error(f"Error: {exc}")
