"""Vẽ ô trạng thái Active Liveness (realtime), dùng chung cho vòng lặp polling
và các bước PASS/FAIL cuối cùng."""

import streamlit as st


def render_active_status(
    status_box,
    instruction,
    status,
    remaining,
    api_status,
    deepfake_info=None,
    hold_progress=None,
):
    color_map = {"DETECTING": "#d29922", "PASS": "#3fb950", "FAIL": "#f85149"}
    color = color_map.get(status, "#9aa4b2")

    deepfake_row = ""
    if deepfake_info is not None:
        n_samples, avg_score, is_suspected = deepfake_info
        if n_samples > 0:
            df_color = "#f85149" if is_suspected else "#3fb950"
            deepfake_row = f"""
            <div class="status-row">
                <span class="status-label">🛡️ Deepfake score (live, {n_samples} lần check): </span>
                <b style="color:{df_color};">{avg_score*100:.1f}%</b>
            </div>
            """
        else:
            deepfake_row = """
            <div class="status-row">
                <span class="status-label">🛡️ Deepfake score (live): </span>
                <b style="color:#9aa4b2;">đang thu thập dữ liệu...</b>
            </div>
            """

    hold_row = ""
    if hold_progress is not None:
        held, required = hold_progress
        hold_row = f"""
        <div class="status-row">
            <span class="status-label">Giữ yên trong oval: </span>
            <b style="color:#3fb950;">{held:.1f}s / {required:.1f}s</b>
        </div>
        """

    with status_box.container():
        st.markdown(f"""
        <div class="status-card">
            <div class="card-title">🔎 Trạng thái Active Liveness (realtime)</div>
            <div class="status-row">
                <span class="status-label">Hướng dẫn: </span>
                <b>{instruction}</b>
            </div>
            <div class="status-row">
                <span class="status-label">Trạng thái: </span>
                <b style="color:{color};">{status}</b>
            </div>
            <div class="status-row">
                <span class="status-label">Thời gian còn lại: </span>
                <b>{remaining:.1f}s</b>
            </div>
            {hold_row}
            {deepfake_row}
            <div class="status-row">
                <span class="status-label">Gọi API Backend: </span>
                <b>{api_status}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
