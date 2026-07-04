# -*- coding: utf-8 -*-
"""
Mo phong Hanh lang dong (Dynamic Priority Corridor)
Ung dung ly thuyet hang doi (M/G/1) trong cau hinh duong CHC dong
tai Cang HKQT Tan Son Nhat.

Giao dien: dashboard nen toi (RunwayFlow-style) + animation HTML/CSS/JS
cho cap duong bang song song 25L / 25R.

Chay ung dung:
    streamlit run app.py
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from simulation import SimConfig, run_simulation

# --------------------------------------------------------------------------- #
# Cau hinh trang
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="RunwayFlow | Mo phong Hanh lang dong",
    page_icon="✈️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# THEME - CSS nen toi kieu dashboard
# --------------------------------------------------------------------------- #
def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root{
            --bg:#0a0e14; --bg-side:#0d1219; --card:#111826; --card-2:#0e141d;
            --border:#212a37; --text:#e6edf3; --muted:#8b96a5;
            --teal:#2dd4bf; --teal-dim:#12332e;
            --cyan:#22d3ee; --orange:#fb923c; --amber:#f5c451;
            --green:#34d399; --red:#f87171; --purple:#8338EC;
        }
        .stApp{ background:var(--bg); color:var(--text); }
        section[data-testid="stSidebar"]{ background:var(--bg-side); border-right:1px solid var(--border); }
        section[data-testid="stSidebar"] *{ color:var(--text); }
        h1,h2,h3,h4,h5,h6, p, span, label, div{ color:var(--text); }
        .stCaption, [data-testid="stCaptionContainer"]{ color:var(--muted) !important; }

        /* metric cards */
        div[data-testid="stMetric"]{
            background:var(--card); border:1px solid var(--border); border-radius:12px;
            padding:14px 16px 10px 16px;
        }
        div[data-testid="stMetricLabel"] p{ color:var(--muted) !important; font-size:0.8rem; }
        div[data-testid="stMetricValue"]{ color:var(--text) !important; }

        /* containers used as cards */
        div[data-testid="stVerticalBlockBorderWrapper"]{
            background:var(--card); border:1px solid var(--border) !important; border-radius:12px;
        }

        /* tabs -> pill toggle */
        button[data-baseweb="tab"]{
            background:var(--card-2); border:1px solid var(--border); border-radius:8px !important;
            margin-right:6px; color:var(--muted) !important; padding:4px 14px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"]{
            background:var(--teal-dim); border-color:var(--teal); color:var(--teal) !important;
        }
        div[data-baseweb="tab-highlight"]{ display:none; }
        div[data-baseweb="tab-border"]{ display:none; }

        /* badges */
        .rf-badge{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem;
                   font-weight:600; letter-spacing:.02em; }
        .rf-badge-ready{ background:#0f2e24; color:var(--green); border:1px solid #1f6f52; }
        .rf-badge-active{ background:#132a3a; color:var(--cyan); border:1px solid #1e5a78; }
        .rf-badge-warn{ background:#3a2410; color:var(--amber); border:1px solid #7a5210; }

        /* section headers */
        .rf-section-title{
            font-size:0.78rem; letter-spacing:.06em; color:var(--muted); font-weight:700;
            text-transform:uppercase; margin:6px 0 10px 0;
        }

        /* top banner */
        .rf-banner{
            display:flex; align-items:center; gap:18px; background:linear-gradient(135deg,#0f2b26,#0d1f1d);
            border:1px solid #1c4a3f; border-radius:14px; padding:16px 22px; margin-bottom:18px;
        }
        .rf-banner-icon{
            width:46px; height:46px; border-radius:10px; background:#123a32; color:var(--teal);
            display:flex; align-items:center; justify-content:center; font-size:1.3rem; flex:none;
        }
        .rf-banner-title{ font-size:1.35rem; font-weight:800; color:#fff; line-height:1.1; }
        .rf-banner-eyebrow{ font-size:0.7rem; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
        .rf-banner-sub{ font-size:0.82rem; color:var(--muted); margin-top:2px; }
        .rf-gauges{ display:flex; gap:26px; margin-left:auto; }
        .rf-gauge{ min-width:170px; }
        .rf-gauge-label{ font-size:0.72rem; color:var(--muted); margin-bottom:4px; }
        .rf-gauge-bar{ width:100%; height:6px; border-radius:4px; background:#132a26; overflow:hidden; }
        .rf-gauge-fill{ height:100%; border-radius:4px; background:var(--cyan); }

        /* runway status card */
        .rf-rwy-head{ display:flex; align-items:center; justify-content:space-between; }
        .rf-rwy-name{ font-weight:800; font-size:1.05rem; }
        .rf-rwy-ops{ color:var(--muted); font-size:0.82rem; margin-top:8px; }

        /* animation widget wrapper (keeps a border like other cards) */
        .rf-anim-card{ border:1px solid var(--border); border-radius:12px; background:var(--card); padding:2px; }

        /* priority / theory cards */
        .rf-code-pill{ background:#0d1420; border:1px solid var(--border); border-radius:8px;
                       padding:6px 10px; font-family:monospace; font-size:0.8rem; color:var(--cyan); }
        .rf-icao-box{ background:#0f2e24; border:1px solid #1f6f52; border-radius:10px; padding:12px 16px;
                      color:#bfe9d8; font-size:0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

MODE_LABELS = {
    "arr_only": "Arrivals Only (chi ha canh)",
    "dep_only": "Departures Only (chi cat canh)",
    "mixed": "Mixed (hon hop - Hanh lang dong)",
}
MODE_KEYS = {v: k for k, v in MODE_LABELS.items()}

MODE_META = {
    "mixed": dict(name="MIXED-MODE", icon="🔁",
                  subtitle="Khai thác hỗn hợp cân bằng trên 2 runway song song"),
    "arr_only": dict(name="ARRIVALS-ONLY", icon="🛬",
                      subtitle="Chỉ khai thác hạ cánh — RWY 25R"),
    "dep_only": dict(name="DEPARTURES-ONLY", icon="🛫",
                       subtitle="Chỉ khai thác cất cánh — RWY 25L"),
}

SCENARIOS = {
    "Tuy chinh": None,
    "Kich ban 1 - Cao diem tiem can bao hoa (lambda=42, 21/21)": dict(
        arrival_rate=21.0, departure_rate=21.0, service_arr=81.8, service_dep=81.8,
    ),
    "Kich ban 2 - Lech pha cung cau (lambda=35, 14/21)": dict(
        arrival_rate=14.0, departure_rate=21.0, service_arr=81.8, service_dep=81.8,
    ),
}

# --------------------------------------------------------------------------- #
# Sidebar - bang dieu khien (bieu mau cau hinh tham so mo phong, muc 4.1.3)
# --------------------------------------------------------------------------- #
st.sidebar.markdown('<div class="rf-section-title">🎛️ Bảng điều khiển</div>', unsafe_allow_html=True)

scenario_name = st.sidebar.selectbox("Kich ban mau (tuy chon)", list(SCENARIOS.keys()))
preset = SCENARIOS[scenario_name]

with st.sidebar.form("sim_form"):
    st.markdown("**Thoi gian mo phong**")
    col_a, col_b = st.columns(2)
    with col_a:
        simulation_time = st.number_input(
            "Sim time (phut)", min_value=5.0, max_value=1440.0, value=60.0, step=5.0)
    with col_b:
        cycle_length = st.number_input(
            "Cycle length (phut)", min_value=0.1, max_value=30.0, value=1.0, step=0.5)

    n_cycles_preview = max(1, round(simulation_time / cycle_length))
    st.caption(f"So chu ky / Cycles = {n_cycles_preview}")

    st.markdown("---")
    st.markdown("**Luu luong tau bay (Poisson)**")
    col_c, col_d = st.columns(2)
    with col_c:
        arrival_rate = st.number_input(
            "Arrival rate \u03bb (chuyen/gio)", min_value=0.0, max_value=200.0,
            value=float(preset["arrival_rate"]) if preset else 21.0, step=1.0)
    with col_d:
        departure_rate = st.number_input(
            "Departure rate \u03bb (chuyen/gio)", min_value=0.0, max_value=200.0,
            value=float(preset["departure_rate"]) if preset else 21.0, step=1.0)

    st.markdown("**Thoi gian phuc vu / ROT (giay)**")
    col_e, col_f = st.columns(2)
    with col_e:
        service_arr = st.number_input(
            "Service (arr) - giay", min_value=1.0, max_value=600.0,
            value=float(preset["service_arr"]) if preset else 75.0, step=1.0)
    with col_f:
        service_dep = st.number_input(
            "Service (dep) - giay", min_value=1.0, max_value=600.0,
            value=float(preset["service_dep"]) if preset else 85.0, step=1.0)

    st.markdown("---")
    st.markdown("**Che do van hanh duong bang**")
    runway_mode_label = st.selectbox("Runway mode", list(MODE_LABELS.values()), index=2)
    runway_mode = MODE_KEYS[runway_mode_label]

    min_hold_time = st.number_input(
        "Minimum hold time (phut) - chi ap dung khi Mixed",
        min_value=0.5, max_value=60.0, value=5.0, step=0.5,
        disabled=(runway_mode != "mixed"),
    )

    st.markdown("---")
    st.markdown("**Tinh huong khan cap**")
    col_g, col_h = st.columns(2)
    with col_g:
        emergency_prob = st.slider(
            "Emergency probability / chu ky", min_value=0.0, max_value=1.0,
            value=0.0, step=0.01)
    with col_h:
        emergency_service_time = st.number_input(
            "Emergency service time (giay)", min_value=10.0, max_value=1800.0,
            value=180.0, step=10.0)

    st.markdown("---")
    compare_policy = st.checkbox(
        "So sanh FCFS (luan phien co dinh) vs Hanh lang dong",
        value=True, disabled=(runway_mode != "mixed"),
        help="Chi co y nghia khi Runway mode = Mixed.",
    )
    seed = st.number_input("Random seed (de tai lap ket qua)", min_value=0, max_value=999999, value=42, step=1)

    anim_speed = st.slider("Tốc độ hoạt cảnh runway", min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("▶️ Chạy mô phỏng", use_container_width=True)

# --------------------------------------------------------------------------- #
# Tieu de
# --------------------------------------------------------------------------- #
st.markdown(
    "<div style='font-size:1.5rem;font-weight:800;'>✈️ RunwayFlow — Mô phỏng Hành lang động</div>"
    "<div style='color:#8b96a5;font-size:0.85rem;margin-bottom:10px;'>"
    "Ứng dụng lý thuyết hàng đợi (M/G/1 · Pollaczek–Khinchine) trong cấu hình đường CHC động "
    "tại Cảng HKQT Tân Sơn Nhất.</div>",
    unsafe_allow_html=True,
)

with st.expander("Giả thiết mô hình (tóm tắt)", expanded=False):
    st.markdown(
        """
- Luong den (**Arrivals**) va luong di (**Departures**) la 2 tien trinh **Poisson** doc lap,
  toc do trung binh \u03bb_arr, \u03bb_dep (chuyen/gio); tong nhu cau \u03bb = \u03bb_arr + \u03bb_dep.
- Thoi gian phuc vu (ROT) co phan phoi tong quat **G**, chi can biet gia tri ky vong va phuong sai
  &rarr; he thong hang doi **M/G/1**.
- Ca hai luong duoc xu ly boi **mot may chu don (single server)** la cum duong CHC.
- Dieu kien on dinh: \u03c1 = \u03bb / \u03bc < 1 (Gia thiet 4).
- Suc chua hang doi vo han (Gia thiet 5).
- Thuat toan **Hanh lang dong**: lien tuc so sanh do dai hang doi hai luong, chu dong nhuong
  quyen uu tien cho luong dang bi don u de tranh bung no phi tuyen tinh cua P-K formula,
  nhung phai duy tri moi che do toi thieu **Minimum Hold Time** de tranh dao chieu lien tuc.
        """
    )

# --------------------------------------------------------------------------- #
# Chay mo phong
# --------------------------------------------------------------------------- #
if "result_dynamic" not in st.session_state:
    st.session_state.result_dynamic = None
    st.session_state.result_fcfs = None
    st.session_state.cfg_used = None

if submitted:
    base_kwargs = dict(
        simulation_time=simulation_time,
        cycle_length=cycle_length,
        arrival_rate=arrival_rate,
        departure_rate=departure_rate,
        service_arr=service_arr,
        service_dep=service_dep,
        runway_mode=runway_mode,
        min_hold_time=min_hold_time,
        emergency_prob=emergency_prob,
        emergency_service_time=emergency_service_time,
        seed=int(seed),
    )
    cfg_dyn = SimConfig(policy="dynamic", **base_kwargs)
    st.session_state.result_dynamic = run_simulation(cfg_dyn)
    st.session_state.cfg_used = cfg_dyn

    if compare_policy and runway_mode == "mixed":
        cfg_fcfs = SimConfig(policy="fcfs", **base_kwargs)
        st.session_state.result_fcfs = run_simulation(cfg_fcfs)
    else:
        st.session_state.result_fcfs = None

result = st.session_state.result_dynamic
result_fcfs = st.session_state.result_fcfs
cfg_used = st.session_state.cfg_used

if result is None:
    st.info("Thiết lập tham số ở thanh bên trái và bấm **Chạy mô phỏng** để bắt đầu.")
    st.stop()

s = result.summary

if s["theory_rho"] is not None and s["theory_rho"] >= 1:
    st.error(
        f"⚠️ Hệ thống KHÔNG ổn định theo lý thuyết (\u03c1 = {s['theory_rho']:.3f} \u2265 1): "
        "tổng nhu cầu vượt quá năng lực phục vụ tối đa. Hàng đợi lý thuyết tiến tới vô cùng "
        "(điểm gãy của công thức P-K). Kết quả mô phỏng bên dưới chỉ phản ánh trạng thái quá tải tạm thời."
    )

# --------------------------------------------------------------------------- #
# BANNER trang thai che do khai thac hien tai
# --------------------------------------------------------------------------- #
meta = MODE_META[cfg_used.runway_mode]
last_lq_arr = result.lq_arr[-1] if len(result.lq_arr) else 0
last_lq_dep = result.lq_dep[-1] if len(result.lq_dep) else 0
max_lq_arr = max(max(result.lq_arr), 1) if len(result.lq_arr) else 1
max_lq_dep = max(max(result.lq_dep), 1) if len(result.lq_dep) else 1
pct_arr = min(100, round(100 * last_lq_arr / max_lq_arr))
pct_dep = min(100, round(100 * last_lq_dep / max_lq_dep))

st.markdown(
    f"""
    <div class="rf-banner">
        <div class="rf-banner-icon">{meta['icon']}</div>
        <div>
            <div class="rf-banner-eyebrow">Chế độ khai thác hiện tại</div>
            <div class="rf-banner-title">{meta['name']}</div>
            <div class="rf-banner-sub">{meta['subtitle']}</div>
        </div>
        <div class="rf-gauges">
            <div class="rf-gauge">
                <div class="rf-gauge-label">Arrival queue &nbsp; {int(last_lq_arr)} / {int(max_lq_arr)}</div>
                <div class="rf-gauge-bar"><div class="rf-gauge-fill" style="width:{pct_arr}%;background:#22d3ee;"></div></div>
            </div>
            <div class="rf-gauge">
                <div class="rf-gauge-label">Departure queue &nbsp; {int(last_lq_dep)} / {int(max_lq_dep)}</div>
                <div class="rf-gauge-bar"><div class="rf-gauge-fill" style="width:{pct_dep}%;background:#fb923c;"></div></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Chỉ số tổng hợp (KPI)</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Throughput", f"{s['throughput_per_h']:.0f} ops/h")
k2.metric("Tổng thao tác", f"{s['cum_served']} ops")
k3.metric("Delay arrival TB", f"{s['wq_total_min']:.1f} phút")
k4.metric("Delay departure TB", f"{s['wq_total_min']:.1f} phút")
k5.metric("Đã hạ cánh", f"{s['served_arr_total']} ac")
k6.metric("Đã cất cánh", f"{s['served_dep_total']} ac")

st.markdown("#### Đối chiếu với lý thuyết hàng đợi M/G/1 (Pollaczek–Khinchine)")
t1, t2, t3, t4 = st.columns(4)
rho_txt = f"{s['theory_rho']:.3f}" if s["theory_rho"] is not None else "N/A"
lq_txt = f"{s['theory_Lq']:.2f}" if s["theory_Lq"] is not None else "\u221e (khong on dinh)"
wq_txt = f"{s['theory_Wq_min']:.2f}" if s["theory_Wq_min"] is not None else "\u221e (khong on dinh)"
t1.metric("\u03c1 lý thuyết (\u03bb/\u03bc)", rho_txt)
t2.metric("Lq lý thuyết (P-K)", lq_txt,
          delta=None if s["theory_Lq"] is None else f"{s['avg_lq_total'] - s['theory_Lq']:+.2f} (mp - lt)")
t3.metric("Wq lý thuyết (phút)", wq_txt,
          delta=None if s["theory_Wq_min"] is None else f"{s['wq_total_min'] - s['theory_Wq_min']:+.2f} (mp - lt)")
t4.metric("\u03bc lý thuyết (chuyến/giờ)", f"{s['mu_theoretical']:.1f}")

st.divider()

# --------------------------------------------------------------------------- #
# RUNWAY SONG SONG - trang thai + ANIMATION HTML/CSS/JS
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Runway song song &nbsp;·&nbsp; Occupancy thời gian thực</div>',
            unsafe_allow_html=True)

# --- chuan bi du lieu khung hinh cho animation (downsample neu qua nhieu) ---
n_total = len(result.cycles)
MAX_FRAMES = 400
if n_total > MAX_FRAMES:
    idxs = np.linspace(0, n_total - 1, MAX_FRAMES).astype(int)
else:
    idxs = np.arange(n_total)

frames = []
cum_arr, cum_dep = 0, 0
served_arr_list = getattr(result, "served_arr", None)
served_dep_list = getattr(result, "served_dep", None)
for i in idxs:
    i = int(i)
    if served_arr_list is not None:
        cum_arr = int(sum(served_arr_list[: i + 1]))
    if served_dep_list is not None:
        cum_dep = int(sum(served_dep_list[: i + 1]))
    frames.append(dict(
        t=round(float(result.time_min[i]), 2),
        mode=result.mode_active[i],
        emergency=bool(result.emergency[i]),
        qa=int(result.lq_arr[i]),
        qd=int(result.lq_dep[i]),
        ca=cum_arr,
        cd=cum_dep,
    ))

frames_json = json.dumps(frames)
base_interval = int(1300 - anim_speed * 110)  # ms giua 2 khung hinh

st.markdown('<div class="rf-anim-card">', unsafe_allow_html=True)
components.html(
    f"""
    <div id="rf-root" style="font-family:'Segoe UI',sans-serif;background:#111826;color:#e6edf3;
         border-radius:12px;padding:14px 18px 18px 18px;">

      <div style="display:flex;gap:14px;margin-bottom:16px;flex-wrap:wrap;">
        <div style="flex:1;min-width:220px;background:#0e141d;border:1px solid #212a37;border-radius:10px;padding:12px 16px;">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span style="font-weight:800;font-size:1.02rem;">RWY 25R</span>
            <span id="hdr-r-badge" style="padding:2px 10px;border-radius:999px;font-size:0.72rem;font-weight:700;
                  background:#0f2e24;color:#34d399;border:1px solid #1f6f52;">SẴN SÀNG</span>
          </div>
          <div id="hdr-r-ops" style="color:#8b96a5;font-size:0.82rem;margin-top:8px;">0 ops · ưu tiên Hạ cánh</div>
          <div id="hdr-r-queue" style="color:#8b96a5;font-size:0.82rem;margin-top:4px;">Hàng đợi hiện tại: 0 tàu</div>
        </div>
        <div style="flex:1;min-width:220px;background:#0e141d;border:1px solid #212a37;border-radius:10px;padding:12px 16px;">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span style="font-weight:800;font-size:1.02rem;">RWY 25L</span>
            <span id="hdr-l-badge" style="padding:2px 10px;border-radius:999px;font-size:0.72rem;font-weight:700;
                  background:#0f2e24;color:#34d399;border:1px solid #1f6f52;">SẴN SÀNG</span>
          </div>
          <div id="hdr-l-ops" style="color:#8b96a5;font-size:0.82rem;margin-top:8px;">0 ops · ưu tiên Cất cánh</div>
          <div id="hdr-l-queue" style="color:#8b96a5;font-size:0.82rem;margin-top:4px;">Hàng đợi hiện tại: 0 tàu</div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap;">
        <button id="rf-play" style="background:#123a32;color:#2dd4bf;border:1px solid #1c4a3f;
                border-radius:8px;padding:6px 14px;cursor:pointer;font-weight:600;">⏸️ Tạm dừng</button>
        <span style="font-size:0.75rem;color:#8b96a5;">Tốc độ hoạt cảnh: điều chỉnh ở sidebar</span>
        <span id="rf-clock" style="margin-left:auto;font-family:monospace;color:#8b96a5;font-size:0.85rem;">T+00:00</span>
        <span id="rf-mode-tag" style="background:#132a3a;color:#22d3ee;border:1px solid #1e5a78;
              border-radius:999px;padding:2px 12px;font-size:0.72rem;font-weight:700;">MIXED</span>
      </div>

      <div id="rf-emergency" style="display:none;background:#3a1414;color:#f87171;border:1px solid #7a1f1f;
           border-radius:8px;padding:6px 12px;margin-bottom:10px;font-size:0.8rem;font-weight:700;
           text-align:center;">🚨 TÌNH HUỐNG KHẨN CẤP — ưu tiên xử lý ngay</div>

      <div style="position:relative;height:56px;margin-bottom:26px;overflow:visible;">
        <div style="position:absolute;left:0;right:0;top:20px;height:16px;background:#232b36;border-radius:3px;
             background-image:repeating-linear-gradient(90deg,#3a4452 0 24px, transparent 24px 48px);"></div>
        <div style="position:absolute;left:6px;top:2px;font-size:0.7rem;color:#8b96a5;">THR 25R · Hạ cánh</div>
        <div style="position:absolute;right:6px;top:2px;font-size:0.7rem;color:#8b96a5;" id="queue-r">Chờ trên không: 0</div>
        <div id="track-r" style="position:absolute;left:0;right:0;top:10px;height:32px;overflow:visible;"></div>
      </div>

      <div style="position:relative;height:56px;overflow:visible;">
        <div style="position:absolute;left:0;right:0;top:20px;height:16px;background:#232b36;border-radius:3px;
             background-image:repeating-linear-gradient(90deg,#3a4452 0 24px, transparent 24px 48px);"></div>
        <div style="position:absolute;left:6px;top:2px;font-size:0.7rem;color:#8b96a5;">THR 25L · Cất cánh</div>
        <div style="position:absolute;right:6px;top:2px;font-size:0.7rem;color:#8b96a5;" id="queue-l">Chờ mặt đất: 0</div>
        <div id="track-l" style="position:absolute;left:0;right:0;top:10px;height:32px;overflow:visible;"></div>
      </div>
    </div>

    <style>
      .rf-plane {{ position:absolute; top:0; font-size:22px; line-height:1; z-index:5;
                   transition-property:left,opacity; transition-timing-function:linear;
                   filter:drop-shadow(0 0 2px rgba(0,0,0,.6)); }}
      .rf-plane-arr {{ transform:scaleX(-1); }}
      .rf-plane-dep {{ transform:scaleX(-1); }}
    </style>

    <script>
      const FRAMES = {frames_json};
      const BASE_INTERVAL = {base_interval};
      let idx = 0;
      let playing = true;
      let timer = null;

      const trackR = document.getElementById('track-r');
      const trackL = document.getElementById('track-l');
      const queueR = document.getElementById('queue-r');
      const queueL = document.getElementById('queue-l');
      const clock = document.getElementById('rf-clock');
      const modeTag = document.getElementById('rf-mode-tag');
      const emerBox = document.getElementById('rf-emergency');
      const playBtn = document.getElementById('rf-play');
      const hdrRBadge = document.getElementById('hdr-r-badge');
      const hdrLBadge = document.getElementById('hdr-l-badge');
      const hdrROps = document.getElementById('hdr-r-ops');
      const hdrLOps = document.getElementById('hdr-l-ops');
      const hdrRQueue = document.getElementById('hdr-r-queue');
      const hdrLQueue = document.getElementById('hdr-l-queue');

      function setBadge(el, active) {{
        if (active) {{
          el.textContent = 'ĐANG HOẠT ĐỘNG';
          el.style.background = '#132a3a'; el.style.color = '#22d3ee'; el.style.borderColor = '#1e5a78';
        }} else {{
          el.textContent = 'SẴN SÀNG';
          el.style.background = '#0f2e24'; el.style.color = '#34d399'; el.style.borderColor = '#1f6f52';
        }}
      }}

      function spawnPlane(track, emoji, colorClass) {{
        const el = document.createElement('div');
        el.className = 'rf-plane ' + colorClass;
        el.textContent = emoji;
        el.style.left = '100%';
        el.style.opacity = '1';
        track.appendChild(el);
        requestAnimationFrame(() => {{
          el.style.transitionDuration = (BASE_INTERVAL * 3.4 / 1000) + 's';
          el.style.left = '-4%';
        }});
        setTimeout(() => {{ el.style.opacity = '0'; }}, BASE_INTERVAL * 3.0);
        setTimeout(() => {{ el.remove(); }}, BASE_INTERVAL * 3.6);
      }}

      function renderFrame(f) {{
        clock.textContent = 'T+' + String(Math.floor(f.t / 60)).padStart(2, '0') + ':' +
                             String(Math.floor(f.t % 60)).padStart(2, '0');
        const modeLabel = {{ arr: 'ARRIVALS', dep: 'DEPARTURES', mixed: 'MIXED' }}[f.mode] || f.mode.toUpperCase();
        modeTag.textContent = modeLabel;
        const modeColors = {{ arr: ['#132a3a', '#22d3ee', '#1e5a78'],
                               dep: ['#3a2410', '#fb923c', '#7a5210'],
                               mixed: ['#123a32', '#2dd4bf', '#1c4a3f'] }};
        const c = modeColors[f.mode] || modeColors.mixed;
        modeTag.style.background = c[0]; modeTag.style.color = c[1]; modeTag.style.borderColor = c[2];

        queueR.textContent = 'Chờ trên không: ' + f.qa;
        queueL.textContent = 'Chờ mặt đất: ' + f.qd;

        emerBox.style.display = f.emergency ? 'block' : 'none';

        const rActive = (f.mode === 'arr' || f.mode === 'mixed');
        const lActive = (f.mode === 'dep' || f.mode === 'mixed');
        setBadge(hdrRBadge, rActive);
        setBadge(hdrLBadge, lActive);
        hdrROps.textContent = f.ca + ' ops · ưu tiên Hạ cánh';
        hdrLOps.textContent = f.cd + ' ops · ưu tiên Cất cánh';
        hdrRQueue.textContent = 'Hàng đợi hiện tại: ' + f.qa + ' tàu';
        hdrLQueue.textContent = 'Hàng đợi hiện tại: ' + f.qd + ' tàu';

        if (rActive) {{ spawnPlane(trackR, '✈️', 'rf-plane-arr'); }}
        if (lActive) {{ spawnPlane(trackL, '🛫', 'rf-plane-dep'); }}
      }}

      function tick() {{
        if (!FRAMES.length) return;
        renderFrame(FRAMES[idx]);
        idx = (idx + 1) % FRAMES.length;
      }}

      function start() {{
        tick();
        timer = setInterval(tick, BASE_INTERVAL);
      }}

      playBtn.addEventListener('click', () => {{
        playing = !playing;
        if (playing) {{
          playBtn.textContent = '⏸️ Tạm dừng';
          start();
        }} else {{
          playBtn.textContent = '▶️ Phát hoạt cảnh';
          clearInterval(timer);
        }}
      }});

      start();
    </script>
    """,
    height=430,
)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------- #
# HANG DOI THOI GIAN THUC (cards)
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Hàng đợi thời gian thực</div>', unsafe_allow_html=True)
qc1, qc2 = st.columns(2)
with qc1:
    with st.container(border=True):
        st.markdown(
            f"""<div class="rf-rwy-head"><span class="rf-rwy-name">🛬 Arrival</span>
                <span class="rf-badge rf-badge-active">{int(last_lq_arr)}</span></div>""",
            unsafe_allow_html=True,
        )
        if last_lq_arr == 0:
            st.caption("Hàng đợi trống")
        else:
            st.caption(f"{int(last_lq_arr)} tàu bay đang chờ hạ cánh")
with qc2:
    with st.container(border=True):
        st.markdown(
            f"""<div class="rf-rwy-head"><span class="rf-rwy-name">🛫 Departure</span>
                <span class="rf-badge rf-badge-warn">{int(last_lq_dep)}</span></div>""",
            unsafe_allow_html=True,
        )
        if last_lq_dep == 0:
            st.caption("Hàng đợi trống")
        else:
            st.caption(f"{int(last_lq_dep)} tàu bay đang chờ cất cánh")

st.divider()

# --------------------------------------------------------------------------- #
# DIEN BIEN THEO THOI GIAN - tabs (thay cho 3 bieu do roi rac)
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Diễn biến theo thời gian</div>', unsafe_allow_html=True)

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3"),
)

tab_q, tab_u, tab_g = st.tabs(["📉 Hàng đợi", "⚙️ Mức sử dụng & Throughput", "🎯 Định hướng ưu tiên"])

with tab_q:
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=result.time_min, y=result.lq_arr, name="Lq - Arrivals",
                                mode="lines", line=dict(color="#22d3ee")))
    fig_q.add_trace(go.Scatter(x=result.time_min, y=result.lq_dep, name="Lq - Departures",
                                mode="lines", line=dict(color="#fb923c")))
    total_lq = [a + d for a, d in zip(result.lq_arr, result.lq_dep)]
    fig_q.add_trace(go.Scatter(x=result.time_min, y=total_lq, name="Lq - Total",
                                mode="lines", line=dict(color="#8b96a5", dash="dot")))
    if s["theory_Lq"] is not None:
        fig_q.add_hline(y=s["theory_Lq"], line_dash="dash", line_color="#34d399",
                         annotation_text="Lq lý thuyết (P-K)", annotation_position="top left")
    fig_q.update_layout(xaxis_title="Thời gian (phút)", yaxis_title="Số tàu bay trong hàng đợi",
                         height=380, legend=dict(orientation="h", y=1.12), **PLOTLY_DARK)
    st.plotly_chart(fig_q, use_container_width=True)

with tab_u:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        fig_u = go.Figure()
        fig_u.add_trace(go.Scatter(x=result.time_min, y=result.utilization, name="\u03c1 (chu kỳ)",
                                    mode="lines", line=dict(color="#2dd4bf"), fill="tozeroy"))
        if s["theory_rho"] is not None:
            fig_u.add_hline(y=s["theory_rho"], line_dash="dash", line_color="#f87171",
                             annotation_text="\u03c1 lý thuyết")
        fig_u.update_layout(xaxis_title="Thời gian (phút)", yaxis_title="Mức sử dụng (0-1)",
                             height=340, yaxis_range=[0, 1.05], **PLOTLY_DARK)
        st.plotly_chart(fig_u, use_container_width=True)
    with col_right:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Đã phục vụ - Arrivals", "Đã phục vụ - Departures"],
            values=[s["served_arr_total"], s["served_dep_total"]],
            marker=dict(colors=["#22d3ee", "#fb923c"]),
            hole=0.45,
        )])
        fig_pie.update_layout(height=340, showlegend=True, **PLOTLY_DARK)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab_g:
    if cfg_used.runway_mode == "mixed":
        mode_numeric = [1 if m == "arr" else 0 for m in result.mode_active]
        fig_gantt = go.Figure()
        fig_gantt.add_trace(go.Scatter(
            x=result.time_min, y=mode_numeric, mode="lines", line_shape="hv",
            line=dict(color="#8338EC", width=2), name="Hướng ưu tiên",
            fill="tozeroy",
        ))
        emergency_x = [t for t, e in zip(result.time_min, result.emergency) if e]
        if emergency_x:
            fig_gantt.add_trace(go.Scatter(
                x=emergency_x, y=[1.05] * len(emergency_x), mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#f87171"),
                name="Sự kiện khẩn cấp",
            ))
        fig_gantt.update_layout(
            xaxis_title="Thời gian (phút)",
            yaxis=dict(tickvals=[0, 1], ticktext=["Ưu tiên Departures", "Ưu tiên Arrivals"], range=[-0.1, 1.2]),
            height=280, **PLOTLY_DARK,
        )
        st.plotly_chart(fig_gantt, use_container_width=True)
    else:
        st.caption("Biểu đồ định hướng ưu tiên chỉ áp dụng cho chế độ Mixed.")

st.divider()

# --------------------------------------------------------------------------- #
# PHAN TICH LY THUYET HANG DOI M/G/1 (cards)
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Phân tích lý thuyết hàng đợi M/G/1</div>', unsafe_allow_html=True)
st.caption(
    "Mỗi runway được mô hình hoá như một server M/G/1: máy bay đến theo phân phối Poisson (\u03bb), "
    "thời gian chiếm dụng theo phân phối tổng quát với E[S] và Cv. Lq = \u03bb\u00b7Wq là thời gian chờ "
    "trung bình — cơ sở để thuật toán quyết định khi nào chuyển ưu tiên."
)

theory_arr, theory_dep = st.columns(2)
with theory_arr:
    with st.container(border=True):
        st.markdown(f"**Luồng Arrival** &nbsp; <span style='color:#8b96a5;'>(\u03bb={cfg_used.arrival_rate:.2f}/h)</span>",
                    unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Chỉ số": ["\u03c1 (utilization)", "Lq (ac chờ)", "Wq (phút chờ)"],
            "Lý thuyết": [rho_txt, lq_txt, wq_txt],
            "Mô phỏng": [f"{s['avg_utilization']:.2f}", f"{s['avg_lq_total']:.2f}", f"{s['wq_total_min']:.2f}"],
        }))
with theory_dep:
    with st.container(border=True):
        st.markdown(f"**Luồng Departure** &nbsp; <span style='color:#8b96a5;'>(\u03bb={cfg_used.departure_rate:.2f}/h)</span>",
                    unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Chỉ số": ["\u03c1 (utilization)", "Lq (ac chờ)", "Wq (phút chờ)"],
            "Lý thuyết": [rho_txt, lq_txt, wq_txt],
            "Mô phỏng": [f"{s['avg_utilization']:.2f}", f"{s['avg_lq_total']:.2f}", f"{s['wq_total_min']:.2f}"],
        }))

st.divider()

# --------------------------------------------------------------------------- #
# THUAT TOAN CHUYEN DOI UU TIEN (cards mo ta logic Hanh lang dong)
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Thuật toán chuyển đổi ưu tiên (Dynamic Corridor)</div>',
            unsafe_allow_html=True)
pc1, pc2, pc3 = st.columns(3)
with pc1:
    with st.container(border=True):
        st.markdown("**🛬 Điều kiện 1 — Ưu tiên Arrival**")
        st.markdown('<div class="rf-code-pill">if Lq_arr(t) > Lq_dep(t)</div>', unsafe_allow_html=True)
        st.caption("Khi hàng đợi hạ cánh dài hơn hàng đợi cất cánh, hệ thống nhường quyền ưu tiên "
                   "cho luồng Arrivals để tránh máy bay trên không phải chờ/bay vòng (holding, go-around).")
with pc2:
    with st.container(border=True):
        st.markdown("**🛫 Điều kiện 2 — Ưu tiên Departure**")
        st.markdown('<div class="rf-code-pill">if Lq_dep(t) > Lq_arr(t)</div>', unsafe_allow_html=True)
        st.caption("Khi hàng đợi cất cánh dài hơn, ưu tiên chuyển sang Departures để giải phóng "
                   "tắc nghẽn dưới đất (taxiway, apron) và tránh làm tăng delay dây chuyền.")
with pc3:
    with st.container(border=True):
        st.markdown("**⏱️ Ràng buộc — Minimum Hold Time**")
        st.markdown(f'<div class="rf-code-pill">hold ≥ {cfg_used.min_hold_time:.1f} phút</div>',
                    unsafe_allow_html=True)
        st.caption("Mỗi lần chuyển hướng ưu tiên phải duy trì tối thiểu Minimum Hold Time để tránh "
                   "đảo chiều liên tục (thrashing), giữ ổn định vận hành thực tế.")

st.markdown(
    """
    <div class="rf-icao-box">
    ✅ <b>Ràng buộc an toàn ICAO:</b> mọi thao tác trên cùng runway phải tuân thủ phân cách tối thiểu
    (wake turbulence separation) giữa các chuyến. Khi cả hai hàng đợi cùng vượt ngưỡng, hệ thống ưu tiên
    dựa trên chỉ số Wq = Lq/\u03bb theo lý thuyết hàng đợi M/G/1.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# So sanh FCFS vs Hanh lang dong
# --------------------------------------------------------------------------- #
if result_fcfs is not None:
    st.markdown('<div class="rf-section-title">So sánh: FCFS (luân phiên cố định) vs Hành lang động</div>',
                unsafe_allow_html=True)
    sf = result_fcfs.summary
    compare_df = pd.DataFrame({
        "Chỉ số": ["Lq trung bình (tàu bay)", "Wq trung bình (phút)", "Hệ số sử dụng \u03c1",
                   "Throughput (chuyến/giờ)", "Số chuyến đã phục vụ"],
        "FCFS (cố định 1:1)": [sf["avg_lq_total"], sf["wq_total_min"], sf["avg_utilization"],
                                sf["throughput_per_h"], sf["cum_served"]],
        "Hành lang động": [s["avg_lq_total"], s["wq_total_min"], s["avg_utilization"],
                           s["throughput_per_h"], s["cum_served"]],
    })
    compare_df["Cải thiện (%)"] = np.where(
        compare_df["FCFS (cố định 1:1)"] != 0,
        (compare_df["FCFS (cố định 1:1)"] - compare_df["Hành lang động"]) / compare_df["FCFS (cố định 1:1)"] * 100,
        0.0,
    )

    col_i, col_j = st.columns([1, 1])
    with col_i:
        st.dataframe(compare_df.style.format({
            "FCFS (cố định 1:1)": "{:.2f}", "Hành lang động": "{:.2f}", "Cải thiện (%)": "{:+.1f}%"
        }), use_container_width=True, hide_index=True)
    with col_j:
        fig_cmp = go.Figure()
        metrics_for_bar = ["Lq trung bình (tàu bay)", "Wq trung bình (phút)"]
        sub = compare_df[compare_df["Chỉ số"].isin(metrics_for_bar)]
        fig_cmp.add_trace(go.Bar(name="FCFS", x=sub["Chỉ số"], y=sub["FCFS (cố định 1:1)"], marker_color="#fb923c"))
        fig_cmp.add_trace(go.Bar(name="Hành lang động", x=sub["Chỉ số"], y=sub["Hành lang động"], marker_color="#22d3ee"))
        fig_cmp.update_layout(barmode="group", height=340, legend=dict(orientation="h", y=1.15), **PLOTLY_DARK)
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.caption(
        "Giá trị Cải thiện (%) dương nghĩa là Hành lang động giúp giảm chỉ số đó so với FCFS "
        "(ví dụ Wq giảm = giảm thời gian chờ trung bình)."
    )
    st.divider()

# --------------------------------------------------------------------------- #
# Bang du lieu chi tiet + tai xuong CSV
# --------------------------------------------------------------------------- #
st.markdown('<div class="rf-section-title">Bảng dữ liệu chi tiết theo chu kỳ</div>', unsafe_allow_html=True)
df = pd.DataFrame(result.to_records())
df = df.rename(columns={
    "cycles": "Chu kỳ", "time_min": "Thời gian (phút)", "lq_arr": "Lq Arrivals",
    "lq_dep": "Lq Departures", "served_arr": "Đã phục vụ (arr)", "served_dep": "Đã phục vụ (dep)",
    "mode_active": "Hướng ưu tiên", "utilization": "Hệ số sử dụng", "emergency": "Khẩn cấp",
})
st.dataframe(df, use_container_width=True, height=320)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Tải xuống kết quả (CSV)", data=csv_bytes,
    file_name="ket_qua_mo_phong_hanh_lang_dong.csv", mime="text/csv",
)

st.caption(
    "Mô hình mô phỏng được xây dựng dựa trên đề tài 'Ứng dụng lý thuyết hàng đợi trong cấu hình "
    "đường cất hạ cánh động tại Cảng HKQT Tân Sơn Nhất' - Học viện Hàng không Việt Nam, 2026."
)
