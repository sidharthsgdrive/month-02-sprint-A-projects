import os

import requests
import streamlit as st


def get_backend_url():
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return os.environ.get("BACKEND_URL", "http://localhost:8000")


BACKEND_URL = get_backend_url()

st.set_page_config(page_title="Emoji Search", page_icon="🔍")
st.title("🔍 Emoji Semantic Search")

query = st.text_input("Describe the emoji you're looking for", placeholder="e.g. happy, birthday cake, sad")

if query:
    try:
        response = requests.get(f"{BACKEND_URL}/search", params={"q": query, "top_k": 10})
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as e:
        st.error(f"Could not reach backend: {e}")
        results = []

    cols = st.columns(5)
    for i, item in enumerate(results):
        with cols[i % 5]:
            st.markdown(f"<div style='font-size:48px; text-align:center'>{item['emoji']}</div>", unsafe_allow_html=True)
            st.caption(item["name"])
