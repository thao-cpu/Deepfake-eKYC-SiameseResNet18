"""Header của trang + badge trạng thái hệ thống (SYSTEM READY)."""

import streamlit as st


def render_header():
    st.markdown("""
    <div class="header">
        <div class="title">🔐 Liveness Detection</div>
        <div class="subtitle">
            AI-powered real/fake verification system
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_system_status():
    col_status, col_space = st.columns([1, 3])

    with col_status:
        st.markdown(
            '<span class="ready">● SYSTEM READY</span>',
            unsafe_allow_html=True
        )

    st.divider()
