import os
import csv
from datetime import datetime

import streamlit as st
import pandas as pd

import tsce_demo

RATINGS_FILE = "ratings.csv"

st.title("TSCE Comparison Chat")

prompt = st.text_input("Enter your prompt", key="prompt_input")

if st.button("Submit", key="submit_prompt") and prompt:
    baseline = tsce_demo._chat([
        {"role": "system", "content": tsce_demo.sys_prompt_inject},
        {"role": "user", "content": prompt},
    ])
    _, tsce_answer = tsce_demo.tsce(prompt)
    st.session_state["baseline"] = baseline
    st.session_state["tsce_answer"] = tsce_answer
    st.session_state["current_prompt"] = prompt

if "baseline" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        st.write(st.session_state["baseline"])
    with col2:
        st.subheader("TSCE")
        st.write(st.session_state["tsce_answer"])

    rating = st.radio(
        "Which response is better?",
        ["Baseline better", "TSCE better", "Tie"],
        key="rating_choice",
    )
    if st.button("Submit Rating", key="submit_rating"):
        file_exists = os.path.exists(RATINGS_FILE)
        with open(RATINGS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "prompt", "rating"])
            writer.writerow([
                datetime.now().isoformat(),
                st.session_state.get("current_prompt", ""),
                rating,
            ])
        st.success("Rating saved.")

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

