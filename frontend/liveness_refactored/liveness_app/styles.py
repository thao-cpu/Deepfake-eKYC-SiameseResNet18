"""CSS tuỳ chỉnh cho toàn bộ giao diện Streamlit của app."""

import streamlit as st

CUSTOM_CSS = """
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .header {
        padding: 10px 0 25px 0;
    }

    .title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9aa4b2;
        font-size: 1rem;
    }

    .status-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 22px;
        min-height: 210px;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .status-row {
        padding: 10px 0;
        border-bottom: 1px solid #252b33;
    }

    .status-label {
        color: #9aa4b2;
    }

    .ready {
        color: #3fb950;
        font-weight: 600;
    }

    .waiting {
        color: #d29922;
        font-weight: 600;
    }

    .info-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        margin-top: 20px;
    }

    .result-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
</style>
"""


def inject_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
