"""共用前置處理 —— 練習 A（睡眠報告）和練習 B（YASA 比對）都從這裡取資料。

負責三件事：

1. 把 Sleep-EDF 的 hypnogram annotation 展開成固定長度（30 秒）的 epoch 階段陣列，
   並依 AASM 把 stage 3 / 4 合併成 N3。
2. 把超出實際訊號長度的標註 clip 掉（這筆資料的標註跨到 24 小時，
   比 23.58 小時的訊號還長）。
3. 裁切出真正的 TIB 區間 —— 不裁切的話睡眠效率會算成 ~25%。

沒有任何相依套件超出現有 `.venv`（只用 numpy + mne）。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

# --------------------------------------------------------------------------
# 常數
# --------------------------------------------------------------------------
EPOCH_SEC = 30.0

# AASM 五類的整數編碼；-1 保留給無效 epoch（Movement time / Sleep stage ?）
W, N1, N2, N3, R = 0, 1, 2, 3, 4
INVALID = -1

STAGE_LABELS = ("W", "N1", "N2", "N3", "R")
STAGE_CODES = (W, N1, N2, N3, R)

# hypnogram 由上而下的繪圖順序：W → R → N1 → N2 → N3
PLOT_ORDER = (W, R, N1, N2, N3)

# 和 app.py 同一套色系：W / R 各自一個色相，N1→N3 同一藍色相由淺到深
STAGE_COLORS = {
    W: "#e34948",
    N1: "#5598e7",
    N2: "#2a78d6",
    N3: "#1c5cab",
    R: "#4a3aa7",
    INVALID: "#8a8a85",
}

# Sleep-EDF 的原始標籤 -> 五類編碼。stage 4 併進 N3 是 AASM 的規定。
_ANN_TO_STAGE = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage N1": N1,
    "Sleep stage 2": N2,
    "Sleep stage N2": N2,
    "Sleep stage 3": N3,
    "Sleep stage N3": N3,
    "Sleep stage 4": N3,
    "Sleep stage N4": N3,
    "Sleep stage R": R,
    "Sleep stage REM": R,
    "Movement time": INVALID,
    "Sleep stage ?": INVALID,
}

# 裁切時在第一個/最後一個睡眠 epoch 外各保留的時間
TRIM_PAD_SEC = 30 * 60.0

# 覺醒次數只計算長度 >= 這個秒數的 W 區段
MIN_AWAKENING_SEC = 30.0

# 分期只能用真正 100 Hz 取樣的電生理頻道。
# Resp / EMG / Temp 原始是 1 Hz，被 MNE 上採樣成階梯訊號，
# 拿去做頻譜或分期都會得到假結果。
ANALYSIS_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
EEG_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
STEPPED_CHANNELS = ("Resp oro-nasal", "EMG submental", "Temp rectal")

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PSG = DATA_DIR / "SC4002E0-PSG.edf"
DEFAULT_HYPNO = DATA_DIR / "SC4002EC-Hypnogram.edf"


# --------------------------------------------------------------------------
# 讀檔
# --------------------------------------------------------------------------
def read_raw(psg_path: Path | str = DEFAULT_PSG, preload: bool = False):
    """讀 PSG。preload=False 時只讀 header，很快。"""
    return mne.io.read_raw_edf(str(psg_path), preload=preload)


def n_epochs_for(raw, epoch_sec: float = EPOCH_SEC) -> int:
    """訊號放得下幾個完整 epoch（不足一個 epoch 的尾巴直接丟掉）。"""
    return int(raw.n_times / float(raw.info["sfreq"]) // epoch_sec)


def stages_from_hypnogram(
    hypno_path: Path | str,
    n_epochs: int,
    epoch_sec: float = EPOCH_SEC,
) -> tuple[np.ndarray, dict]:
    """把 annotation 展開成長度 n_epochs 的階段陣列（int），並回報 clip 掉多少。

    每個 epoch 取「中點落在哪個 annotation」來決定階段。這筆資料的
    annotation 是連續的、長度都是 30 秒的整數倍，所以中點法不會有歧義。
    沒有任何 annotation 覆蓋到的 epoch 一律標成 INVALID。
    """
    ann = mne.read_annotations(str(hypno_path))
    onset = np.asarray(ann.onset, dtype=float)
    duration = np.asarray(ann.duration, dtype=float)
    codes = np.array(
        [_ANN_TO_STAGE.get(str(d), INVALID) for d in ann.description], dtype=int
    )

    order = np.argsort(onset, kind="stable")
    onset, duration, codes = onset[order], duration[order], codes[order]
    end = onset + duration

    centers = (np.arange(n_epochs) + 0.5) * epoch_sec
    idx = np.searchsorted(onset, centers, side="right") - 1
    covered = (idx >= 0) & (centers < end[np.clip(idx, 0, len(end) - 1)])

    stages = np.full(n_epochs, INVALID, dtype=int)
    stages[covered] = codes[idx[covered]]

    signal_sec = n_epochs * epoch_sec
    info = {
        "annotation_span_sec": float(end[-1]) if len(end) else 0.0,
        "signal_sec": signal_sec,
        "clipped_sec": max(0.0, float(end[-1]) - signal_sec) if len(end) else 0.0,
        "uncovered_epochs": int((~covered).sum()),
    }
    return stages, info


# --------------------------------------------------------------------------
# 裁切
# --------------------------------------------------------------------------
def trim_indices(
    stages: np.ndarray,
    epoch_sec: float = EPOCH_SEC,
    pad_sec: float = TRIM_PAD_SEC,
) -> tuple[int, int]:
    """回傳 TIB 區間的 [start, stop] epoch 索引（含兩端）。

    規則：找第一個和最後一個「非 W 的有效」epoch，往前/往後各保留 pad_sec。
    無效 epoch（Movement time / Sleep stage ?）不能當端點，否則裁切點會被
    資料尾端那段 `Sleep stage ?` 拉走。
    """
    sleep = (stages != W) & (stages != INVALID)
    if not sleep.any():
        raise ValueError("整段錄音找不到任何有效的睡眠 epoch，無法決定 TIB 區間")

    pad = int(round(pad_sec / epoch_sec))
    first, last = int(np.argmax(sleep)), int(len(sleep) - 1 - np.argmax(sleep[::-1]))
    return max(0, first - pad), min(len(stages) - 1, last + pad)


def stage_runs(stages: np.ndarray) -> list[tuple[int, int, int]]:
    """把階段陣列壓成 [(stage, start_idx, stop_idx_exclusive), ...]。"""
    if len(stages) == 0:
        return []
    edges = np.flatnonzero(np.diff(stages)) + 1
    bounds = np.concatenate(([0], edges, [len(stages)]))
    return [
        (int(stages[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])
    ]


# --------------------------------------------------------------------------
# 指標
# --------------------------------------------------------------------------
def sleep_metrics(stages: np.ndarray, epoch_sec: float = EPOCH_SEC) -> dict:
    """對「已裁切」的階段陣列算七項指標。時間單位一律是秒。

    WASO 定義為「入睡點之後、TIB 區間內的所有 W」，包含最後一段睡醒後的 W，
    這樣才會滿足 TIB = SOL + TST + WASO + 無效時間（誤差只來自無效 epoch）。
    """
    stages = np.asarray(stages, dtype=int)
    n = len(stages)
    valid = stages != INVALID
    is_sleep = valid & (stages != W)

    tib = n * epoch_sec
    tst = float(is_sleep.sum()) * epoch_sec

    onset = int(np.argmax(is_sleep)) if is_sleep.any() else n
    sol = onset * epoch_sec

    rem_hits = np.flatnonzero(stages == R)
    rem_latency = float(rem_hits[0] - onset) * epoch_sec if len(rem_hits) else float("nan")

    after = stages[onset:]
    waso = float((after == W).sum()) * epoch_sec

    # 覺醒次數：入睡後的 W 區段，扣掉最後那段睡醒後的 W
    min_epochs = int(np.ceil(MIN_AWAKENING_SEC / epoch_sec))
    awakenings = [
        (a, b)
        for stage, a, b in stage_runs(after)
        if stage == W and b < len(after) and (b - a) >= min_epochs
    ]

    per_stage = {
        code: float((stages == code).sum()) * epoch_sec for code in STAGE_CODES
    }
    invalid_sec = float((~valid).sum()) * epoch_sec

    return {
        "n_epochs": n,
        "epoch_sec": epoch_sec,
        "tib_sec": tib,
        "tst_sec": tst,
        "sleep_efficiency": tst / tib if tib else float("nan"),
        "sol_sec": sol,
        "onset_epoch": onset,
        "rem_latency_sec": rem_latency,
        "waso_sec": waso,
        "n_awakenings": len(awakenings),
        "awakening_runs": awakenings,
        "per_stage_sec": per_stage,
        "invalid_sec": invalid_sec,
        "n_invalid": int((~valid).sum()),
        "balance_error_sec": tib - (sol + tst + waso + invalid_sec),
    }


# --------------------------------------------------------------------------
# 時間軸
# --------------------------------------------------------------------------
def clock_datetimes(
    meas_date, offsets_sec: np.ndarray
) -> list[_dt.datetime]:
    """把「錄音起點起算的秒數」換成實際時鐘時間。

    Sleep-EDF 的 meas_date 帶 UTC tzinfo，但它其實是當地時間，
    所以這裡把 tzinfo 拿掉直接當本地時鐘用，畫出來才是 22:00 而不是位移過的值。
    """
    if meas_date is None:
        base = _dt.datetime(2000, 1, 1)
    else:
        base = meas_date.replace(tzinfo=None)
    return [base + _dt.timedelta(seconds=float(s)) for s in offsets_sec]


def fmt_hms(seconds: float) -> str:
    """3930 -> '1h 05m'；小於一小時就只給分鐘。"""
    if seconds != seconds:  # NaN
        return "n/a"
    total_min = int(round(seconds / 60.0))
    h, m = divmod(total_min, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


# --------------------------------------------------------------------------
# 一次到位
# --------------------------------------------------------------------------
@dataclass
class Recording:
    """一筆錄音的完整前置處理結果。A 和 B 都拿這個當唯一入口。"""

    psg_path: Path
    hypno_path: Path
    raw: object
    meas_date: object
    epoch_sec: float
    stages_full: np.ndarray  # 整段錄音（已 clip 到訊號長度）
    trim_start: int
    trim_stop: int  # 含
    clip_info: dict

    @property
    def stages(self) -> np.ndarray:
        """裁切後的階段陣列 —— 練習 A 的所有統計都用這個。"""
        return self.stages_full[self.trim_start : self.trim_stop + 1]

    @property
    def trim_offset_sec(self) -> float:
        return self.trim_start * self.epoch_sec

    def epoch_offsets(self, trimmed: bool = True) -> np.ndarray:
        """每個 epoch 起點距離錄音開始的秒數。"""
        n = len(self.stages) if trimmed else len(self.stages_full)
        base = self.trim_offset_sec if trimmed else 0.0
        return base + np.arange(n) * self.epoch_sec

    def describe_trim(self) -> str:
        full_h = len(self.stages_full) * self.epoch_sec / 3600.0
        trim_h = len(self.stages) * self.epoch_sec / 3600.0
        clock = clock_datetimes(
            self.meas_date,
            np.array([self.trim_start * self.epoch_sec,
                      (self.trim_stop + 1) * self.epoch_sec]),
        )
        return (
            f"裁切前：epoch 0–{len(self.stages_full) - 1}"
            f"（{len(self.stages_full)} 個，{full_h:.2f} 小時）\n"
            f"裁切後：epoch {self.trim_start}–{self.trim_stop}"
            f"（{len(self.stages)} 個，{trim_h:.2f} 小時）\n"
            f"對應時鐘時間：{clock[0]:%H:%M} → {clock[1]:%H:%M}"
        )


def load_recording(
    psg_path: Path | str = DEFAULT_PSG,
    hypno_path: Path | str = DEFAULT_HYPNO,
    epoch_sec: float = EPOCH_SEC,
    preload: bool = False,
) -> Recording:
    """讀 EDF + hypnogram，展開成 epoch、clip、算出裁切區間。"""
    psg_path, hypno_path = Path(psg_path), Path(hypno_path)
    raw = read_raw(psg_path, preload=preload)
    n_epochs = n_epochs_for(raw, epoch_sec)
    stages_full, clip_info = stages_from_hypnogram(hypno_path, n_epochs, epoch_sec)
    start, stop = trim_indices(stages_full, epoch_sec)
    return Recording(
        psg_path=psg_path,
        hypno_path=hypno_path,
        raw=raw,
        meas_date=raw.info.get("meas_date"),
        epoch_sec=epoch_sec,
        stages_full=stages_full,
        trim_start=start,
        trim_stop=stop,
        clip_info=clip_info,
    )
