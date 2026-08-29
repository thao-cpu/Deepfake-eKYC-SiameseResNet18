"""
Entrypoint của app Liveness Detection.

File này CHỈ điều phối thứ tự vẽ UI — toàn bộ logic chi tiết nằm trong
package `liveness_app/`:
    config.py             hằng số dùng chung (URL backend, ngưỡng, ...)
    styles.py              CSS tuỳ chỉnh
    header.py               header trang + badge SYSTEM READY
    passive_liveness.py     upload ảnh/video + gọi backend passive
    deepfake_filters.py      2 hàm biến đổi ảnh giả lập artifact deepfake
    liveness_processor.py    class LivenessProcessor (video processor)
    virtual_camera.py        phát hiện virtual camera
    render_status.py         vẽ ô trạng thái Active Liveness realtime
    active_backend.py        gọi backend xác minh Active Liveness + kết luận eKYC
    active_liveness.py       orchestrator khối Active Liveness (webrtc, polling...)

Chạy bằng: streamlit run app.py
"""

from liveness_app import config, styles, header
from liveness_app.passive_liveness import render_passive_liveness
from liveness_app.active_liveness import render_active_liveness

config.configure_page()
styles.inject_custom_css()

header.render_header()
header.render_system_status()

render_passive_liveness()
render_active_liveness()
