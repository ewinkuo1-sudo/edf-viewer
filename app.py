"""EDF Viewer - 用 Streamlit + MNE + Plotly 瀏覽 EDF 訊號波形。"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import mne
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

mne.set_log_level("ERROR")

DATA_DIR = Path(__file__).parent / "data"

INK = "#0b0b0b"
MUTED = "#8a8a85"
SECONDARY = "#52514e"
GRID = "#ececea"

# 每個頻道各自一個 facet、標題就在波形正上方，所以顏色只是輔助掃視。
# 這組順序通過 adjacent-pair 檢查（最差 CVD ΔE 8.4、normal-vision ΔE 19.3）。
CHANNEL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# W / REM 各自一個色相，N1→N4 是同一藍色相由淺到深（睡眠深度是有序的）
STAGE_COLORS = {
    "W": "#e34948",
    "N1": "#5598e7",
    "N2": "#2a78d6",
    "N3": "#1c5cab",
    "N4": "#0d366b",
    "REM": "#4a3aa7",
    "Movement": SECONDARY,
    "?": MUTED,
}

STAGE_NAMES = {
    "W": "清醒 Wake",
    "N1": "淺睡 N1",
    "N2": "淺睡 N2",
    "N3": "深睡 N3",
    "N4": "深睡 N4",
    "REM": "快速動眼 REM",
    "Movement": "體動 Movement",
    "?": "未標註",
}

# mne.read_annotations() 讀 Sleep-EDF Hypnogram 的原始標籤 -> 簡短階段名
STAGE_ALIASES = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage N1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage N2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage N3": "N3",
    "Sleep stage 4": "N4",
    "Sleep stage N4": "N4",
    "Sleep stage R": "REM",
    "Sleep stage REM": "REM",
    "Movement time": "Movement",
    "Sleep stage ?": "?",
}

JUMP_STAGES = ["W", "N1", "N2", "N3", "REM"]

# MNE 的 EDF reader 只把這些原始單位換算成伏特儲存；其餘頻道維持原始物理量。
# 注意不能用 raw.get_channel_types() 判斷 —— Sleep-EDF 的每個頻道都被標成 'eeg'，
# 連 Temp rectal（°C）也是，照 type 換算會讓非電生理頻道的數值爆掉 1e6 倍。
VOLT_UNITS = {"µV", "μV", "uV", "mV", "V"}

st.set_page_config(page_title="EDF Viewer", layout="wide")

st.markdown(
    """
    <style>
      div.block-container {padding-top: 2.2rem; padding-bottom: 1rem;}
      div[data-testid="stSidebarUserContent"] {padding-top: 1.5rem;}
      /* 導覽列的按鈕高度對齊 number_input */
      div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
          white-space: nowrap;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# 載入
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="讀取 EDF...")
def load_raw(path: str):
    return mne.io.read_raw_edf(path, preload=False)


@st.cache_data(show_spinner=False)
def cache_upload(name: str, payload: bytes) -> str:
    digest = hashlib.md5(payload).hexdigest()[:12]
    tmp = Path(tempfile.gettempdir()) / "edf_viewer_uploads"
    tmp.mkdir(exist_ok=True)
    dest = tmp / f"{digest}-{name}"
    dest.write_bytes(payload)
    return str(dest)


def find_hypnogram(psg_path: Path) -> Path | None:
    """SC4002E0-PSG.edf -> data/SC4002EC-Hypnogram.edf（前 6 碼相同即視為同一段錄製）。"""
    if not DATA_DIR.is_dir():
        return None
    key = psg_path.stem.split("-")[0][:6].upper()
    for cand in sorted(DATA_DIR.glob("*.edf")):
        if "hypnogram" in cand.stem.lower() and cand.stem.split("-")[0][:6].upper() == key:
            return cand
    return None


@st.cache_data(show_spinner=False)
def load_epoch_stages(path: str, epoch_len: float, n_epochs: int) -> list[str | None]:
    """每個 epoch 中點落在哪個睡眠階段；沒有標註的 epoch 為 None。"""
    ann = mne.read_annotations(path)
    onsets = np.asarray(ann.onset, dtype=float)
    ends = onsets + np.asarray(ann.duration, dtype=float)
    labels = [STAGE_ALIASES.get(str(d), "?") for d in ann.description]

    order = np.argsort(onsets)
    onsets, ends = onsets[order], ends[order]
    labels = [labels[i] for i in order]

    centers = (np.arange(n_epochs) + 0.5) * epoch_len
    idx = np.searchsorted(onsets, centers, side="right") - 1
    return [
        labels[i] if i >= 0 and centers[k] < ends[i] else None
        for k, i in enumerate(idx)
    ]


def channel_unit(raw, ch: str) -> tuple[float, str]:
    """回傳 (放大倍率, 顯示單位)。"""
    orig = (getattr(raw, "_orig_units", None) or {}).get(ch, "n/a")
    if orig in VOLT_UNITS:
        return 1e6, "µV"
    return 1.0, "" if orig in ("n/a", "", None) else str(orig)


def sep() -> str:
    return f"<span style='color:#d5d5d0;margin:0 0.75rem'>|</span>"


# --------------------------------------------------------------------------
# 檔案選擇（sidebar）
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.25rem;font-weight:700;color:{INK};"
        f"margin-bottom:0.8rem'>檔案選擇</div>",
        unsafe_allow_html=True,
    )
    source = st.radio("資料來源", ["從 data/ 目錄選擇", "上傳檔案"], index=0)

    existing = sorted(
        (p for p in DATA_DIR.glob("*.edf") if "hypnogram" not in p.stem.lower()),
    ) if DATA_DIR.is_dir() else []

    picked, uploaded = None, None
    if source == "從 data/ 目錄選擇":
        if existing:
            picked = st.selectbox("選擇 EDF 檔案", [p.name for p in existing])
        else:
            st.caption(f"`{DATA_DIR}` 裡沒有 .edf 檔案。")
    else:
        uploaded = st.file_uploader("上傳 EDF 檔案", type=["edf"])

if uploaded is not None:
    edf_path = Path(cache_upload(uploaded.name, uploaded.getvalue()))
    display_name = uploaded.name
elif picked:
    edf_path = DATA_DIR / picked
    display_name = picked
else:
    st.info(f"請上傳一個 .edf 檔案，或把檔案放進 `{DATA_DIR}` 目錄後重新整理。")
    st.stop()

try:
    raw = load_raw(str(edf_path))
except Exception as exc:  # noqa: BLE001
    st.error(f"無法讀取 {display_name}：{exc}")
    st.stop()

sfreq = float(raw.info["sfreq"])
n_times = raw.n_times
total_seconds = n_times / sfreq
meas_date = raw.info.get("meas_date")
meas_txt = meas_date.strftime("%Y-%m-%d %H:%M:%S") if meas_date else "錄製時間未知"

with st.sidebar:
    st.markdown(f"<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    channels = st.multiselect(
        "頻道", raw.ch_names, default=raw.ch_names[: min(8, len(raw.ch_names))]
    )
    epoch_len = float(st.slider("Epoch 長度（秒）", 5, 300, 30, step=5))

n_epochs = max(1, int(np.ceil(total_seconds / epoch_len)))

# --------------------------------------------------------------------------
# 頂部資訊列
# --------------------------------------------------------------------------
st.markdown(
    f"<div style='font-size:0.88rem;color:{SECONDARY};margin-bottom:1rem'>"
    f"<b style='color:{INK}'>{display_name}</b>{sep()}{len(raw.ch_names)} ch"
    f"{sep()}{sfreq:g} Hz{sep()}{total_seconds / 3600:.1f} hr{sep()}{meas_txt}"
    f"</div>",
    unsafe_allow_html=True,
)

# 換檔或改 epoch 長度時重設索引，避免超出範圍（epoch 對外是 1-based）
signature = (str(edf_path), epoch_len)
if st.session_state.get("_signature") != signature:
    st.session_state["_signature"] = signature
    st.session_state["epoch"] = 1
st.session_state["epoch"] = int(np.clip(st.session_state.get("epoch", 1), 1, n_epochs))

hypno_path = find_hypnogram(edf_path)
epoch_stages = (
    load_epoch_stages(str(hypno_path), epoch_len, n_epochs) if hypno_path else []
)


def shift_epoch(delta: int) -> None:
    st.session_state["epoch"] = int(np.clip(st.session_state["epoch"] + delta, 1, n_epochs))


def jump_to_stage(target: str) -> None:
    current = st.session_state["epoch"] - 1
    for idx in list(range(current + 1, n_epochs)) + list(range(0, current + 1)):
        if epoch_stages[idx] == target:
            st.session_state["epoch"] = idx + 1
            return
    st.toast(f"找不到 {target} 階段的 epoch")


# --------------------------------------------------------------------------
# 導覽列（全部同一行）
# --------------------------------------------------------------------------
nav = st.columns(
    [1.15, 1, 1, 1.15, 0.35, 2.3, 0.35, 1, 1, 1, 1, 1], vertical_alignment="bottom"
)
nav[0].button("⏮ -100", on_click=shift_epoch, args=(-100,), width="stretch")
nav[1].button("◀ -1", on_click=shift_epoch, args=(-1,), width="stretch")
nav[2].button("+1 ▶", on_click=shift_epoch, args=(1,), width="stretch")
nav[3].button("+100 ⏭", on_click=shift_epoch, args=(100,), width="stretch")
nav[5].number_input(
    "epoch", min_value=1, max_value=n_epochs, step=1, key="epoch", label_visibility="collapsed"
)
for col, stage in zip(nav[7:], JUMP_STAGES):
    col.button(
        f"▸ {stage}",
        on_click=jump_to_stage,
        args=(stage,),
        width="stretch",
        disabled=not epoch_stages,
        help="下一個此睡眠階段的 epoch" if epoch_stages else "找不到對應的 Hypnogram 檔案",
    )

epoch_idx = st.session_state["epoch"] - 1
t_start = epoch_idx * epoch_len
t_stop = min(t_start + epoch_len, total_seconds)

# --------------------------------------------------------------------------
# 睡眠階段
# --------------------------------------------------------------------------
meta = (
    f"Epoch {epoch_idx + 1}/{n_epochs}{sep()}{t_start:g}s — {t_stop:g}s"
    if epoch_stages
    else f"Epoch {epoch_idx + 1}/{n_epochs}{sep()}{t_start:g}s — {t_stop:g}s"
    f"{sep()}無對應的 Hypnogram"
)
if epoch_stages:
    stage = epoch_stages[epoch_idx] or "?"
    color = STAGE_COLORS.get(stage, MUTED)
    badge = (
        f"<span style='font-size:1.7rem;font-weight:800;color:{color};"
        f"line-height:1'>{stage}</span>"
        f"<span style='font-size:1rem;font-weight:600;color:{color};"
        f"margin-left:0.6rem'>{STAGE_NAMES.get(stage, stage)}</span>"
    )
else:
    badge = ""
st.markdown(
    f"<div style='display:flex;align-items:baseline;margin:0.9rem 0 0.2rem'>{badge}"
    f"<span style='font-size:0.85rem;color:{MUTED};margin-left:"
    f"{'0.9rem' if badge else '0'}'>{meta}</span></div>",
    unsafe_allow_html=True,
)

if not channels:
    st.warning("請在左側至少選擇一個頻道。")
    st.stop()

# --------------------------------------------------------------------------
# 波形圖
# --------------------------------------------------------------------------
start_sample = int(round(t_start * sfreq))
stop_sample = min(int(round(t_stop * sfreq)), n_times)
data = raw.get_data(picks=channels, start=start_sample, stop=stop_sample)
times = start_sample / sfreq + np.arange(data.shape[1]) / sfreq

units, scaled = [], []
for ch, row in zip(channels, data):
    factor, unit = channel_unit(raw, ch)
    scaled.append(row * factor)
    units.append(unit)

fig = make_subplots(
    rows=len(channels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=min(0.04, 0.9 / max(1, len(channels))),
    subplot_titles=channels,
)
for i, (ch, row, unit) in enumerate(zip(channels, scaled, units), start=1):
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=row,
            mode="lines",
            name=ch,
            line=dict(width=1, color=CHANNEL_COLORS[(i - 1) % len(CHANNEL_COLORS)]),
            hovertemplate=f"%{{x:.3f}}s<br>%{{y:.2f}} {unit}<extra>{ch}</extra>",
        ),
        row=i,
        col=1,
    )
    fig.update_yaxes(
        title_text=unit or None,
        title_font=dict(size=11, color=MUTED),
        title_standoff=8,
        tickfont=dict(size=10, color=MUTED),
        gridcolor=GRID,
        zeroline=False,
        showline=False,
        ticks="",
        row=i,
        col=1,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", row=i, col=1)

fig.update_xaxes(
    title_text="時間（秒）",
    title_font=dict(size=11, color=MUTED),
    tickfont=dict(size=10, color=MUTED),
    row=len(channels),
    col=1,
)
fig.update_annotations(font=dict(size=13, color=SECONDARY))
fig.update_layout(
    height=max(360, 155 * len(channels)),
    margin=dict(l=80, r=30, t=30, b=45),
    showlegend=False,
    hovermode="x unified",
    dragmode="pan",
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})

# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------
with st.expander("頻道統計（目前 epoch）", expanded=False):
    st.dataframe(
        {
            "頻道": channels,
            "單位": [u or "-" for u in units],
            "最大值": [float(np.max(r)) for r in scaled],
            "最小值": [float(np.min(r)) for r in scaled],
            "平均值": [float(np.mean(r)) for r in scaled],
            "標準差": [float(np.std(r)) for r in scaled],
        },
        width="stretch",
        hide_index=True,
    )
