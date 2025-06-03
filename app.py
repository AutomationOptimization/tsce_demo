import os
import csv as pycsv
from datetime import datetime

import streamlit as st
import pandas as pd
import pdfplumber
import openai


# -------------------------------------------------------------------
# Compatibility shim: make tsce_chat look like the old tsce_demo API
# -------------------------------------------------------------------
from tsce_agent_demo import tsce_chat as tsce_demo

# Expose variables expected by legacy code
import os

tsce_demo.MODEL = None
tsce_demo.API_TYPE = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
tsce_demo.AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
tsce_demo.AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
tsce_demo.AZURE_DEPLOYMENT = None
tsce_demo.ENDPOINT = ""
tsce_demo.sys_prompt_inject = tsce_demo.DEFAULT_FINAL_PREFIX

def _make_instance():
    dep = tsce_demo.AZURE_DEPLOYMENT if tsce_demo.API_TYPE == "azure" else None
    return tsce_demo.TSCEChat(model=tsce_demo.MODEL, deployment_id=dep)

def tsce(prompt: str):
    """Return (anchor, answer) tuple using the TSCE method."""
    reply = _make_instance()(prompt)
    return reply.anchor, reply.content

def _chat(messages):
    """Plain chat completion – returns assistant message content only."""
    resp = _make_instance()._completion(messages)
    return resp["choices"][0]["message"]["content"]

# Patch into the module namespace for transparent access
tsce_demo.tsce = tsce
tsce_demo._chat = _chat


MODEL_DEPLOYMENTS = {
    "gpt-3.5-turbo":  os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT35",  "gpt35-deploy"),
    "gpt-4o-mini":    os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O",   "gpt4o-mini-deploy"),
    "gpt-4.1-mini":   os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT41",  "gpt41-mini-deploy"),
}

# Let user pick which models to run
st.sidebar.header("Compare models")
selected_models = st.sidebar.multiselect(
    "Which models?", 
    list(MODEL_DEPLOYMENTS.keys()), 
    default=list(MODEL_DEPLOYMENTS.keys())
)
RATINGS_FILE = "ratings.csv"

def chunk_text(text, max_length=13000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

st.title("TSCE Comparison Chat")

# --- File Upload (PDF or CSV) ---
uploaded_file = st.file_uploader("Upload a PDF or CSV for analysis (optional)", type=["pdf", "csv"])
file_text = ""
file_type = None
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        file_type = "pdf"
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                file_text += page.extract_text() or ""
        st.success(f"PDF uploaded. {len(file_text)} characters extracted.")
    elif uploaded_file.type in ["text/csv", "application/vnd.ms-excel", "application/csv"]:
        file_type = "csv"
        df = pd.read_csv(uploaded_file)
        file_text = df.to_csv(index=False)
        st.success(f"CSV uploaded. {len(file_text)} characters extracted.")

prompt = st.text_area(
    "Enter your prompt (required)",
    key="prompt_input",
    height=150  # adjust height as needed
)

if st.button("Submit", key="submit_prompt") and prompt:
    combined_input = prompt
    if file_text:
        combined_input += "\n\n" + file_text

    st.session_state["current_prompt"] = prompt
    st.session_state["file_text"] = file_text
    st.session_state["file_type"] = file_type
    st.session_state["combined_input"] = combined_input

    results = []
    # Determine which model to TSCE (smallest by arbitrary rank)
    size_rank = {"gpt-3.5-turbo": 0, "gpt-4o-mini": 1, "gpt-4.1-mini": 2}
    if len(selected_models) > 1:
        tsce_model = min(selected_models, key=lambda m: size_rank.get(m, float("inf")))
    else:
        tsce_model = selected_models[0]

    # keep it alive for later reruns (e.g. when rating buttons fire)
    st.session_state["tsce_model"] = tsce_model

    chunks = chunk_text(combined_input) if len(combined_input) > 130000 else [combined_input]
    for chunk_idx, chunk in enumerate(chunks, 1):
        chunk_results = {"chunk_idx": chunk_idx, "chunk": chunk, "outputs": {}}
        for model_name in selected_models:
            tsce_demo.MODEL = model_name  # keep legacy var up to date
            if tsce_demo.API_TYPE == "azure":
                dep = MODEL_DEPLOYMENTS[model_name]
                tsce_demo.AZURE_DEPLOYMENT = dep
                tsce_demo.ENDPOINT = (
                    f"{tsce_demo.AZURE_ENDPOINT}/openai/deployments/{dep}/chat/completions"
                    f"?api-version={tsce_demo.AZURE_VERSION}"
                )
            # --- Baseline (always) ---
            base = tsce_demo._chat([
                {"role": "user",   "content": chunk},
            ])

            # --- TSCE (only for the chosen tsce_model) ---
            tsce_out = None
            anchor_val = None
            answer_val = None
            if model_name == tsce_model:
                anchor_val, answer_val = tsce_demo.tsce(chunk)
                tsce_out = answer_val or ""
            chunk_results["outputs"][model_name] = {
                "baseline": base,
                "tsce": tsce_out,
                "tsce_anchor": anchor_val,
                "tsce_answer": answer_val,
            }
        results.append(chunk_results)
    st.session_state["results"] = results

if "results" in st.session_state:
    for result in st.session_state["results"]:
        st.markdown(f"### Chunk {result['chunk_idx']}")
        with st.expander("Show chunk input"):
            st.write(result["chunk"])
        cols = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            out = result["outputs"][model_name]
            with cols[i]:
                st.subheader(model_name)
                st.markdown("**Baseline**")
                st.write(out["baseline"])
                st.markdown("**TSCE**")
                st.write(out["tsce"])
        rating = st.radio(
            f"Which response is better for chunk {result['chunk_idx']}?",
            ["Baseline better", "TSCE better", "Tie"],
            key=f"rating_choice_{result['chunk_idx']}",
        )
        if st.button(f"Submit Rating for chunk {result['chunk_idx']}", key=f"submit_rating_{result['chunk_idx']}"):
            tsce_model = st.session_state.get("tsce_model")
            if tsce_model is None:
                st.error("TSCE model not found – please submit a prompt first.")
                st.stop()
            file_exists = os.path.exists(RATINGS_FILE)
            with open(RATINGS_FILE, "a", newline="") as f:
                writer = pycsv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "prompt", "file_type", "chunk_idx", "chunk_input",
                        "baseline", "tsce_anchor", "tsce_answer", "tsce_full", "rating"
                    ])
                # Grab the outputs for the stored TSCE-processed model
                model_out = result["outputs"].get(tsce_model, {})
                writer.writerow([
                    datetime.now().isoformat(),
                    st.session_state.get("current_prompt", ""),
                    st.session_state.get("file_type", ""),
                    result["chunk_idx"],
                    result["chunk"],
                    model_out.get("baseline", ""),
                    model_out.get("tsce_anchor", ""),
                    model_out.get("tsce_answer", ""),
                    model_out.get("tsce", ""),
                    rating,
                ])
            st.success(f"Rating saved for chunk {result['chunk_idx']}.")

if os.path.exists(RATINGS_FILE):
    df = pd.read_csv(RATINGS_FILE)
    st.subheader("Saved Ratings")
    st.dataframe(df)
    st.download_button(
        "Download ratings.csv",
        df.to_csv(index=False).encode("utf-8"),
        "ratings.csv",
        "text/csv",
    )
