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

STAGE_COLORS = {
    "W": "#F2A93B",
    "N1": "#5BA7D6",
    "N2": "#2E6FA7",
    "N3": "#1F3F7A",
    "N4": "#16305C",
    "REM": "#D2544C",
    "Movement": "#8A8A8A",
    "?": "#8A8A8A",
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


def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


# --------------------------------------------------------------------------
# 選檔（sidebar）
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("檔案")
    existing = sorted(
        p for p in DATA_DIR.glob("*.edf") if "hypnogram" not in p.stem.lower()
    ) if DATA_DIR.is_dir() else []

    picked = st.selectbox(
        "data/ 目錄中的檔案",
        options=["（不選）"] + [p.name for p in existing],
        index=1 if existing else 0,
    )
    uploaded = st.file_uploader("或上傳 .edf 檔案", type=["edf"])

if uploaded is not None:
    edf_path = Path(cache_upload(uploaded.name, uploaded.getvalue()))
    display_name = uploaded.name
elif picked != "（不選）":
    edf_path = DATA_DIR / picked
    display_name = picked
else:
    st.info(f"請從左側上傳一個 .edf 檔案，或把檔案放進 `{DATA_DIR}` 目錄後重新整理。")
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
meas_txt = meas_date.strftime("%Y-%m-%d %H:%M:%S") if meas_date else "未知"

st.caption(
    f"**{display_name}** · {len(raw.ch_names)} 頻道 · {sfreq:g} Hz · "
    f"時長 {fmt_duration(total_seconds)} · 錄製時間 {meas_txt}"
)

# --------------------------------------------------------------------------
# 頻道與 epoch 設定（sidebar）
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("頻道")
    default_channels = raw.ch_names[: min(8, len(raw.ch_names))]
    channels = st.multiselect("要顯示的頻道", raw.ch_names, default=default_channels)

    st.subheader("Epoch")
    epoch_len = st.number_input(
        "Epoch 長度（秒）", min_value=1.0, max_value=300.0, value=30.0, step=5.0
    )

n_epochs = max(1, int(np.ceil(total_seconds / epoch_len)))

# 換檔或改 epoch 長度時重設索引，避免超出範圍
signature = (str(edf_path), float(epoch_len))
if st.session_state.get("_signature") != signature:
    st.session_state["_signature"] = signature
    st.session_state["epoch"] = 0
st.session_state["epoch"] = int(np.clip(st.session_state.get("epoch", 0), 0, n_epochs - 1))

hypno_path = find_hypnogram(edf_path)
epoch_stages = (
    load_epoch_stages(str(hypno_path), float(epoch_len), n_epochs) if hypno_path else []
)


def shift_epoch(delta: int) -> None:
    st.session_state["epoch"] = int(
        np.clip(st.session_state["epoch"] + delta, 0, n_epochs - 1)
    )


def jump_to_stage(target: str) -> None:
    current = st.session_state["epoch"]
    order = list(range(current + 1, n_epochs)) + list(range(0, current + 1))
    for idx in order:
        if epoch_stages[idx] == target:
            st.session_state["epoch"] = idx
            return
    st.toast(f"找不到 {target} 階段的 epoch")


# --------------------------------------------------------------------------
# 導覽列（全部同一行）
# --------------------------------------------------------------------------
nav = st.columns([1, 1, 2.2, 1, 1, 0.5, 1, 1, 1, 1, 1], vertical_alignment="bottom")
nav[0].button("−100", on_click=shift_epoch, args=(-100,), width="stretch")
nav[1].button("−1", on_click=shift_epoch, args=(-1,), width="stretch")
nav[2].number_input(
    f"Epoch (0–{n_epochs - 1})",
    min_value=0,
    max_value=n_epochs - 1,
    step=1,
    key="epoch",
)
nav[3].button("+1", on_click=shift_epoch, args=(1,), width="stretch")
nav[4].button("+100", on_click=shift_epoch, args=(100,), width="stretch")
for col, stage in zip(nav[6:], JUMP_STAGES):
    col.button(
        f"→ {stage}",
        on_click=jump_to_stage,
        args=(stage,),
        width="stretch",
        disabled=not epoch_stages,
        help="下一個此睡眠階段的 epoch" if epoch_stages else "找不到對應的 Hypnogram 檔案",
    )

epoch_idx = st.session_state["epoch"]
t_start = epoch_idx * epoch_len
t_stop = min(t_start + epoch_len, total_seconds)
window_txt = f"{fmt_duration(t_start)} – {fmt_duration(t_stop)}"

if epoch_stages:
    stage_now = epoch_stages[epoch_idx] or "?"
    st.markdown(
        f"<div style='font-size:2.4rem;font-weight:700;line-height:1.1;"
        f"color:{STAGE_COLORS.get(stage_now, STAGE_COLORS['?'])}'>{stage_now}</div>"
        f"<div style='color:#888;font-size:0.8rem;margin-bottom:0.4rem'>"
        f"{window_txt} · {hypno_path.name}</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div style='color:#888;font-size:0.8rem;margin-bottom:0.4rem'>"
        f"{window_txt} · 無 Hypnogram</div>",
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

fig = make_subplots(rows=len(channels), cols=1, shared_xaxes=True, vertical_spacing=0.012)
for i, (ch, row, unit) in enumerate(zip(channels, scaled, units), start=1):
    fig.add_trace(
        go.Scattergl(
            x=times,
            y=row,
            mode="lines",
            name=ch,
            line=dict(width=1, color="#2E6FA7"),
            hovertemplate=f"%{{x:.3f}}s<br>%{{y:.2f}} {unit}<extra>{ch}</extra>",
        ),
        row=i,
        col=1,
    )
    label = f"{ch}<br><span style='font-size:9px;color:#999'>{unit}</span>" if unit else ch
    fig.update_yaxes(
        title_text=label, title_font=dict(size=11), title_standoff=6, row=i, col=1
    )

fig.update_xaxes(title_text="時間（秒）", row=len(channels), col=1)
fig.update_layout(
    height=max(360, 118 * len(channels)),
    margin=dict(l=80, r=20, t=10, b=40),
    showlegend=False,
    hovermode="x unified",
    dragmode="pan",
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
