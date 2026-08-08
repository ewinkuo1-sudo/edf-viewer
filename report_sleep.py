"""練習 A —— 產生自包含的睡眠報告 HTML。

跑法（用現有的 .venv，不需要裝任何東西）：

    .\\.venv\\Scripts\\python.exe report_sleep.py

輸出：report/sleep_report.html（圖表全部 base64 內嵌，單一檔案即可繳交）

注意：matplotlib 在 Windows 畫中文會變成豆腐框，所以所有圖表標籤一律英文，
中文只出現在 HTML 的文字說明裡。
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.signal import welch

import sleep_utils as su

OUT_DIR = Path(__file__).parent / "report"
OUT_HTML = OUT_DIR / "sleep_report.html"

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8a85"
GRID = "#ececea"

# Welch 參數：30 秒 epoch @ 100 Hz = 3000 點。
# nperseg=1024 -> 頻率解析度 ~0.098 Hz，低頻 0.5 Hz 那端才畫得出來。
NPERSEG = 1024
NOVERLAP = 512

BANDS = [
    ("delta", 0.5, 4.0, "#c8d8ef"),
    ("theta", 4.0, 8.0, "#d6ccea"),
    ("alpha", 8.0, 12.0, "#cfe7dd"),
    ("sigma", 12.0, 16.0, "#f3e0c8"),
    ("beta", 16.0, 30.0, "#eed5d5"),
]
PSD_FMIN, PSD_FMAX = 0.5, 30.0

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#c9c9c4",
        "axes.labelcolor": SECONDARY,
        "axes.titlecolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def fig_to_b64(fig, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def img_tag(b64: str, alt: str) -> str:
    return (
        f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
        f'style="width:100%;height:auto;display:block">'
    )


# --------------------------------------------------------------------------
# 圖 1 — Hypnogram
# --------------------------------------------------------------------------
def plot_hypnogram(rec: su.Recording) -> str:
    stages = rec.stages
    offsets = rec.epoch_offsets()
    # 多補一個收尾點，最後一個 epoch 的階梯才畫得完整
    edge_offsets = np.append(offsets, offsets[-1] + rec.epoch_sec)
    times = mdates.date2num(su.clock_datetimes(rec.meas_date, edge_offsets))

    # y 位置：由上而下 W → R → N1 → N2 → N3
    ypos = {code: i for i, code in enumerate(su.PLOT_ORDER)}
    y = np.array([ypos.get(int(s), np.nan) for s in stages], dtype=float)
    y_edge = np.append(y, y[-1])

    fig, ax = plt.subplots(figsize=(14, 4.0))

    # 無效 epoch 先用灰色底標出來
    invalid_label_used = False
    for stage, a, b in su.stage_runs(stages):
        if stage == su.INVALID:
            ax.axvspan(
                times[a],
                times[b],
                color=su.STAGE_COLORS[su.INVALID],
                alpha=0.45,
                zorder=1,
                label=None if invalid_label_used else "Invalid (movement / unscored)",
            )
            invalid_label_used = True

    ax.step(times, y_edge, where="post", color="#3a3a38", lw=1.1, zorder=3)

    # REM 區段用不同顏色蓋上去
    rem_label_used = False
    for stage, a, b in su.stage_runs(stages):
        if stage == su.R:
            seg_x = times[a : b + 1]
            seg_y = np.full(len(seg_x), float(ypos[su.R]))
            ax.step(
                seg_x,
                seg_y,
                where="post",
                color=su.STAGE_COLORS[su.R],
                lw=3.2,
                solid_capstyle="butt",
                zorder=4,
                label=None if rem_label_used else "REM",
            )
            rem_label_used = True

    ax.set_yticks(range(len(su.PLOT_ORDER)))
    ax.set_yticklabels([su.STAGE_LABELS[c] for c in su.PLOT_ORDER])
    ax.set_ylim(len(su.PLOT_ORDER) - 0.5, -0.5)
    ax.set_xlim(times[0], times[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Clock time (HH:MM)")
    ax.set_ylabel("Sleep stage")
    ax.set_title("Hypnogram (trimmed to time in bed)", fontsize=12, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if rem_label_used or invalid_label_used:
        ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    return fig_to_b64(fig)


# --------------------------------------------------------------------------
# 圖 2 — 階段佔比圓餅圖
# --------------------------------------------------------------------------
def plot_stage_pie(metrics: dict) -> str:
    codes = [c for c in su.STAGE_CODES if metrics["per_stage_sec"][c] > 0]
    values = [metrics["per_stage_sec"][c] / 60.0 for c in codes]
    colors = [su.STAGE_COLORS[c] for c in codes]
    labels = [su.STAGE_LABELS[c] for c in codes]

    fig, ax = plt.subplots(figsize=(7, 5.6))
    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.6),
        textprops=dict(fontsize=10),
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontweight("bold")
    ax.legend(
        wedges,
        [f"{lab}  —  {v:.1f} min" for lab, v in zip(labels, values)],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=10,
    )
    ax.set_title("Stage distribution (share of time in bed)", fontsize=12, pad=14)
    return fig_to_b64(fig)


# --------------------------------------------------------------------------
# 圖 3/4 — 各階段功率頻譜
# --------------------------------------------------------------------------
def compute_stage_psd(rec: su.Recording) -> dict:
    """回傳 {channel: {'freqs':…, 'psd':{stage: array}, 'n':{stage: int}}}，單位 µV²/Hz。"""
    sfreq = float(rec.raw.info["sfreq"])
    spe = int(round(rec.epoch_sec * sfreq))  # samples per epoch
    n_ep = len(rec.stages)
    start = int(round(rec.trim_offset_sec * sfreq))

    data = rec.raw.get_data(
        picks=list(su.EEG_CHANNELS), start=start, stop=start + n_ep * spe
    )
    data = data.reshape(len(su.EEG_CHANNELS), n_ep, spe) * 1e6  # V -> µV

    out = {}
    for ci, ch in enumerate(su.EEG_CHANNELS):
        freqs, pxx = welch(
            data[ci], fs=sfreq, nperseg=NPERSEG, noverlap=NOVERLAP, axis=-1
        )
        keep = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
        freqs, pxx = freqs[keep], pxx[:, keep]
        per_stage, counts = {}, {}
        for code in su.STAGE_CODES:
            sel = rec.stages == code
            if sel.any():
                # 先對 epoch 取中位數再平均，避免單一 artifact epoch 主導整條曲線
                per_stage[code] = np.median(pxx[sel], axis=0)
                counts[code] = int(sel.sum())
        out[ch] = {"freqs": freqs, "psd": per_stage, "n": counts}
    return out


def plot_psd(ch: str, block: dict) -> str:
    freqs, psd = block["freqs"], block["psd"]
    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    for name, lo, hi, color in BANDS:
        ax.axvspan(lo, hi, color=color, alpha=0.55, zorder=0, lw=0)
        ax.text(
            np.sqrt(lo * hi),
            0.965,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color=SECONDARY,
        )

    for code in su.STAGE_CODES:
        if code not in psd:
            continue
        ax.loglog(
            freqs,
            psd[code],
            color=su.STAGE_COLORS[code],
            lw=1.7,
            label=f"{su.STAGE_LABELS[code]} (n={block['n'][code]})",
            zorder=3,
        )

    ax.set_xlim(PSD_FMIN, PSD_FMAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD ($\mu V^2$/Hz)")
    ax.set_title(f"Welch power spectral density by stage — {ch}", fontsize=12, pad=10)
    ax.grid(True, which="both", color=GRID, lw=0.7, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    return fig_to_b64(fig)


def band_power_table(psd_blocks: dict) -> dict:
    """每個頻道、每個階段的相對頻帶功率（五個頻帶加總 = 1）。"""
    rel = {}
    for ch, block in psd_blocks.items():
        freqs = block["freqs"]
        rel[ch] = {}
        for code, curve in block["psd"].items():
            powers = []
            for _, lo, hi, _c in BANDS:
                sel = (freqs >= lo) & (freqs < hi)
                powers.append(np.trapezoid(curve[sel], freqs[sel]))
            powers = np.array(powers, dtype=float)
            total = powers.sum()
            rel[ch][code] = powers / total if total > 0 else powers
    return rel


def plot_band_bars(rel: dict) -> str:
    channels = list(rel.keys())
    fig, axes = plt.subplots(1, len(channels), figsize=(11.5, 4.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, ch in zip(axes, channels):
        codes = [c for c in su.STAGE_CODES if c in rel[ch]]
        x = np.arange(len(codes))
        bottom = np.zeros(len(codes))
        for bi, (name, _lo, _hi, color) in enumerate(BANDS):
            vals = np.array([rel[ch][c][bi] for c in codes])
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=name if ax is axes[0] else None,
            )
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.06:
                    ax.text(
                        xi,
                        b + v / 2,
                        f"{v * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=SECONDARY,
                    )
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([su.STAGE_LABELS[c] for c in codes])
        ax.set_ylim(0, 1)
        ax.set_title(ch, fontsize=11)
        ax.set_xlabel("Sleep stage")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Relative band power")
    handles = [Patch(facecolor=c, label=n) for n, _lo, _hi, c in BANDS]
    fig.legend(
        handles=handles,
        loc="center right",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(1.06, 0.5),
    )
    fig.suptitle("Relative band power by stage", fontsize=12, y=1.0)
    fig.tight_layout()
    return fig_to_b64(fig)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
CSS = """
:root { --ink:#0b0b0b; --sec:#52514e; --muted:#8a8a85; --grid:#ececea; --accent:#2a78d6; }
* { box-sizing: border-box; }
body { margin:0; padding:0 0 4rem; background:#f7f7f5; color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft JhengHei",
                   "PingFang TC","Noto Sans TC",Roboto,Helvetica,Arial,sans-serif;
       line-height:1.65; font-size:15px; }
.wrap { max-width:1080px; margin:0 auto; padding:0 1.5rem; }
header { background:#fff; border-bottom:1px solid var(--grid); padding:2.4rem 0 1.9rem;
         margin-bottom:2rem; }
h1 { margin:0 0 .5rem; font-size:1.85rem; letter-spacing:-.01em; }
.sub { color:var(--sec); font-size:.92rem; }
.sub b { color:var(--ink); }
h2 { font-size:1.22rem; margin:2.6rem 0 .3rem; letter-spacing:-.005em; }
h2 .num { color:var(--accent); font-weight:700; margin-right:.5rem; }
.lede { color:var(--sec); font-size:.93rem; margin:.2rem 0 1.1rem; }
.card { background:#fff; border:1px solid var(--grid); border-radius:12px;
        padding:1.3rem 1.5rem; margin-bottom:1.2rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(232px,1fr)); gap:1rem;
        margin:1rem 0 .4rem; }
.metric { background:#fff; border:1px solid var(--grid); border-radius:12px; padding:1.1rem 1.2rem; }
.metric .label { font-size:.78rem; letter-spacing:.06em; text-transform:uppercase;
                 color:var(--muted); font-weight:700; }
.metric .value { font-size:1.9rem; font-weight:750; letter-spacing:-.02em; margin:.35rem 0 .1rem;
                 line-height:1.15; }
.metric .ref { font-size:.82rem; color:var(--muted); }
.metric .note { font-size:.86rem; color:var(--sec); margin-top:.55rem; }
.flag { display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.04em;
        padding:.12rem .5rem; border-radius:999px; vertical-align:.32rem; margin-left:.5rem; }
.ok  { background:#e6f4ec; color:#166b41; }
.warn{ background:#fdf0dc; color:#8a5a08; }
.info{ background:#e9f0fb; color:#1d5297; }
table { border-collapse:collapse; width:100%; font-size:.92rem; margin:.4rem 0 .2rem; }
th, td { padding:.55rem .7rem; text-align:right; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; }
thead th { font-size:.78rem; letter-spacing:.05em; text-transform:uppercase;
           color:var(--muted); border-bottom:1.5px solid #dcdcd8; }
tbody tr:nth-child(odd) { background:#fafaf9; }
tbody tr.total { font-weight:700; background:#f2f5fa; }
tbody tr.total td { border-bottom:none; }
.swatch { display:inline-block; width:.65rem; height:.65rem; border-radius:2px;
          margin-right:.5rem; vertical-align:baseline; }
.figure { margin:.6rem 0 .2rem; }
.cap { font-size:.85rem; color:var(--muted); margin-top:.6rem; }
ul { margin:.4rem 0 .2rem; padding-left:1.15rem; }
li { margin:.28rem 0; }
code { background:#f2f2ef; padding:.1rem .35rem; border-radius:4px; font-size:.88em; }
footer { color:var(--muted); font-size:.84rem; margin-top:3rem; padding-top:1.2rem;
         border-top:1px solid var(--grid); }
"""


def metric_card(label, value, ref, note, flag=None) -> str:
    cls, txt = flag if flag else ("", "")
    badge = f'<span class="flag {cls}">{html.escape(txt)}</span>' if flag else ""
    return f"""
      <div class="metric">
        <div class="label">{html.escape(label)}</div>
        <div class="value">{value}{badge}</div>
        <div class="ref">正常參考：{html.escape(ref)}</div>
        <div class="note">{note}</div>
      </div>"""


def build_html(rec: su.Recording, m: dict, figs: dict, rel: dict) -> str:
    tib_min = m["tib_sec"] / 60.0
    tst_min = m["tst_sec"] / 60.0
    se = m["sleep_efficiency"] * 100.0
    clock = su.clock_datetimes(
        rec.meas_date,
        np.array(
            [
                rec.trim_start * rec.epoch_sec,
                (rec.trim_stop + 1) * rec.epoch_sec,
                0.0,
                len(rec.stages_full) * rec.epoch_sec,
            ]
        ),
    )
    lights_off, lights_on, rec_start, rec_end = clock

    # ---- 指標卡 ----
    se_flag = ("ok", "正常") if se >= 80 else ("warn", "偏低")
    sol_flag = ("info", "受裁切規則限制")
    rem_min = m["rem_latency_sec"] / 60.0
    rem_flag = ("ok", "正常") if 60 <= rem_min <= 130 else ("warn", "偏離")
    waso_min = m["waso_sec"] / 60.0
    waso_flag = ("ok", "正常") if waso_min <= 40 else ("warn", "偏高")
    awk_flag = ("ok", "正常") if m["n_awakenings"] <= 25 else ("warn", "偏多")

    cards = "".join(
        [
            metric_card(
                "Sleep Efficiency",
                f"{se:.1f}%",
                "健康成人 85–95%",
                "睡著的時間佔躺床時間的比例，是整體睡眠品質最常引用的單一數字。",
                se_flag,
            ),
            metric_card(
                "TST（總睡眠時間）",
                su.fmt_hms(m["tst_sec"]),
                "健康成人 7–9 小時",
                "扣掉所有清醒後，真正處在 N1/N2/N3/R 的時間總和。",
                ("ok", "正常") if 6.5 <= tst_min / 60 <= 9.5 else ("warn", "偏離"),
            ),
            metric_card(
                "TIB（躺床時間）",
                su.fmt_hms(m["tib_sec"]),
                "健康成人 7–9.5 小時",
                f"裁切後的分析區間，對應 {lights_off:%H:%M} 到 {lights_on:%H:%M}。",
                ("ok", "正常") if 7 <= tib_min / 60 <= 10 else ("info", "註記"),
            ),
            metric_card(
                "SOL（入睡潛伏期）",
                su.fmt_hms(m["sol_sec"]),
                "健康成人 10–20 分鐘（<30 分視為正常）",
                "從分析區間起點到第一個睡眠 epoch 的時間。<b>這個數字是裁切規則的產物</b>"
                "（規則規定往前保留 30 分鐘），詳見下方「方法與限制」。",
                sol_flag,
            ),
            metric_card(
                "REM Latency",
                su.fmt_hms(m["rem_latency_sec"]),
                "健康成人 70–120 分鐘",
                "從入睡到第一次進入 REM 的時間。太短（<60 分）在文獻上和憂鬱症、"
                "猝睡症有關聯。",
                rem_flag,
            ),
            metric_card(
                "WASO（入睡後清醒）",
                su.fmt_hms(m["waso_sec"]),
                "健康成人 <30 分鐘",
                "入睡之後在分析區間內所有清醒時間的總和（含最後醒來到起床那段）。",
                waso_flag,
            ),
            metric_card(
                "覺醒次數",
                f"{m['n_awakenings']} 次",
                "健康成人 10–20 次（短暫覺醒）",
                "入睡後、睡醒前的連續 W 區段數，只計 ≥30 秒者。",
                awk_flag,
            ),
        ]
    )

    # ---- 階段表 ----
    rows = []
    for code in su.STAGE_CODES:
        sec = m["per_stage_sec"][code]
        pct_tib = sec / m["tib_sec"] * 100 if m["tib_sec"] else 0
        pct_tst = sec / m["tst_sec"] * 100 if (m["tst_sec"] and code != su.W) else None
        rows.append(
            f"<tr><td><span class='swatch' style='background:{su.STAGE_COLORS[code]}'></span>"
            f"{su.STAGE_LABELS[code]}</td>"
            f"<td>{sec / 60:.1f}</td><td>{int(sec / rec.epoch_sec)}</td>"
            f"<td>{pct_tib:.1f}%</td>"
            f"<td>{'—' if pct_tst is None else f'{pct_tst:.1f}%'}</td></tr>"
        )
    if m["n_invalid"]:
        rows.append(
            f"<tr><td><span class='swatch' style='background:{su.STAGE_COLORS[su.INVALID]}'>"
            f"</span>無效（排除）</td><td>{m['invalid_sec'] / 60:.1f}</td>"
            f"<td>{m['n_invalid']}</td>"
            f"<td>{m['invalid_sec'] / m['tib_sec'] * 100:.1f}%</td><td>—</td></tr>"
        )
    rows.append(
        f"<tr class='total'><td>TIB 合計</td><td>{tib_min:.1f}</td>"
        f"<td>{m['n_epochs']}</td><td>100.0%</td>"
        f"<td>TST {tst_min:.1f} 分</td></tr>"
    )
    stage_table = "\n".join(rows)

    # ---- 相對頻帶功率表 ----
    band_rows = []
    for ch in rel:
        for code in su.STAGE_CODES:
            if code not in rel[ch]:
                continue
            vals = "".join(f"<td>{v * 100:.1f}%</td>" for v in rel[ch][code])
            band_rows.append(
                f"<tr><td>{html.escape(ch)}</td><td style='text-align:left'>"
                f"<span class='swatch' style='background:{su.STAGE_COLORS[code]}'></span>"
                f"{su.STAGE_LABELS[code]}</td>{vals}</tr>"
            )
    band_table = "\n".join(band_rows)
    band_head = "".join(f"<th>{n}</th>" for n, _l, _h, _c in BANDS)

    balance = m["balance_error_sec"]
    check_sum = m["sol_sec"] + m["tst_sec"] + m["waso_sec"]

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>睡眠分析報告 — {html.escape(rec.psg_path.name)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1>睡眠分析報告</h1>
    <div class="sub">
      資料：<b>{html.escape(rec.psg_path.name)}</b> ／ 標註：<b>{html.escape(rec.hypno_path.name)}</b><br>
      錄音起訖：{rec_start:%Y-%m-%d %H:%M} → {rec_end:%m-%d %H:%M}
      （{len(rec.stages_full) * rec.epoch_sec / 3600:.2f} 小時）<br>
      分析區間（裁切後 TIB）：<b>{lights_off:%H:%M} → {lights_on:%H:%M}</b>
      （{tib_min / 60:.2f} 小時，epoch {rec.trim_start}–{rec.trim_stop}）
    </div>
  </div>
</header>

<div class="wrap">

<h2><span class="num">1</span>Hypnogram</h2>
<p class="lede">整夜睡眠階段的階梯圖。由上而下是 W → R → N1 → N2 → N3，越往下代表睡得越深。
REM 區段用紫色加粗標出，灰色底則是被排除的無效 epoch。</p>
<div class="card"><div class="figure">{figs['hypnogram']}</div>
<div class="cap">健康成人的典型型態：整夜 4–6 個週期，深睡（N3）集中在前半夜，
REM 段落越接近清晨越長。這筆資料看得出前半夜 N3 明顯較多，符合預期。</div></div>

<h2><span class="num">2</span>睡眠指標總覽</h2>
<p class="lede">七項指標與健康成人參考範圍。每張卡片下方一句白話說明這個數字在講什麼。</p>
<div class="grid">{cards}</div>

<h2><span class="num">3</span>各階段時間與佔比</h2>
<p class="lede">「佔 TIB」是相對整段躺床時間，「佔 TST」是相對總睡眠時間（不含 W）——
臨床上談各期比例時講的是後者。</p>
<div class="card">
  <table>
    <thead><tr><th>階段</th><th>時間（分）</th><th>Epoch 數</th>
      <th>佔 TIB</th><th>佔 TST</th></tr></thead>
    <tbody>{stage_table}</tbody>
  </table>
  <div class="cap">健康成人佔 TST 的參考比例：N1 約 2–5%、N2 約 45–55%、
  N3 約 13–23%、REM 約 20–25%。</div>
</div>
<div class="card"><div class="figure">{figs['pie']}</div></div>

<h2><span class="num">4</span>各階段 EEG 功率頻譜</h2>
<p class="lede">對裁切區間內每個 epoch 做 Welch PSD（nperseg={NPERSEG}、50% overlap），
再依人工標註分組取中位數。雙對數座標，背景色塊是五個頻帶。
只用 <code>EEG Fpz-Cz</code> 和 <code>EEG Pz-Oz</code>——
Resp / EMG / Temp 原始只有 1 Hz，被 MNE 上採樣成階梯訊號，做頻譜會得到假結果。</p>
<div class="card"><div class="figure">{figs['psd_fpz']}</div>
<div class="cap">額葉導程 Fpz-Cz。N3 在 delta（0.5–4 Hz）明顯抬高，
是深睡慢波的直接證據；W 和 R 的低頻最低。</div></div>
<div class="card"><div class="figure">{figs['psd_pz']}</div>
<div class="cap">頂枕導程 Pz-Oz。這個導程對後頭部 alpha（8–12 Hz）敏感，
清醒閉眼時的 alpha 峰在這裡比 Fpz-Cz 清楚。</div></div>

<h2><span class="num">5</span>相對頻帶功率</h2>
<p class="lede">把每個階段的頻譜在五個頻帶上積分後正規化成 100%，看的是「能量分佈的形狀」
而不是絕對大小，所以不受個體振幅差異影響。</p>
<div class="card"><div class="figure">{figs['bands']}</div></div>
<div class="card">
  <table>
    <thead><tr><th>頻道</th><th>階段</th>{band_head}</tr></thead>
    <tbody>{band_table}</tbody>
  </table>
  <div class="cap">判讀重點：N3 的 delta 佔比最高（慢波睡眠）；
  N2 的 sigma（12–16 Hz）相對抬升對應睡眠紡錘波；
  W 與 R 的高頻（beta）佔比高於各非快速動眼期。</div>
</div>

<h2><span class="num">6</span>方法與限制</h2>
<div class="card">
<ul>
  <li><b>Epoch 化</b>：30 秒一個 epoch，取中點落在哪個 annotation 決定階段。
      stage 3 與 stage 4 依 AASM 合併成 N3。</li>
  <li><b>標註 clip</b>：標註總跨度 {rec.clip_info['annotation_span_sec']:.0f} 秒（24 小時），
      實際訊號只有 {rec.clip_info['signal_sec']:.0f} 秒，
      超出的 {rec.clip_info['clipped_sec']:.0f} 秒（尾端的 <code>Sleep stage ?</code>）已整段丟棄。</li>
  <li><b>無效 epoch</b>：<code>Movement time</code> 與 <code>Sleep stage ?</code> 標成 -1，
      不計入任何統計。整段錄音共 {int((rec.stages_full == su.INVALID).sum())} 個，
      落在分析區間內的有 <b>{m['n_invalid']}</b> 個（{m['invalid_sec'] / 60:.1f} 分鐘）。</li>
  <li><b>裁切</b>：原始錄音從 {rec_start:%H:%M} 開始，睡前躺著清醒了 7.24 小時。
      不裁切的話睡眠效率會算成約 25%，完全失真。
      規則是取第一個與最後一個「非 W 的有效」epoch，往前後各留 30 分鐘：
      epoch <b>{rec.trim_start}–{rec.trim_stop}</b>
      （原始 0–{len(rec.stages_full) - 1}），
      {len(rec.stages_full) * rec.epoch_sec / 3600:.2f} 小時 → <b>{tib_min / 60:.2f} 小時</b>。</li>
  <li><b>SOL 的限制</b>：因為裁切規則規定「往前保留 30 分鐘」，
      分析區間起點必然落在第一個睡眠 epoch 前 30 分鐘，
      所以 SOL 恆等於 30 分 00 秒，<b>不是真正量到的入睡潛伏期</b>。
      真正的 SOL 需要 lights-off 時間點，Sleep-EDF 沒有提供這個標記
      （若從錄音起點起算會是 7.24 小時，那是受試者裝好電極後的自由活動時間，同樣沒有意義）。</li>
  <li><b>WASO 定義</b>：本報告的 WASO 涵蓋入睡後到分析區間結束的所有 W，
      包含最後一次醒來到區間結束那段，這樣才會滿足
      TIB = SOL + TST + WASO + 無效時間。
      驗證：{m['sol_sec'] / 60:.1f} + {tst_min:.1f} + {waso_min:.1f}
      = {check_sum / 60:.1f} 分，加上無效 {m['invalid_sec'] / 60:.1f} 分
      = {tib_min:.1f} 分 = TIB，誤差 {abs(balance):.1f} 秒。</li>
</ul>
</div>

<footer>
由 <code>report_sleep.py</code> 產生 ／ 前置處理見 <code>sleep_utils.py</code> ／
所有圖表以 base64 內嵌，本檔案可獨立開啟，不需要網路或外部資源。
</footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("讀取 EDF 與 hypnogram ...")
    rec = su.load_recording(preload=False)
    print(rec.describe_trim())
    print(
        f"標註 clip：跨度 {rec.clip_info['annotation_span_sec']:.0f}s -> "
        f"訊號 {rec.clip_info['signal_sec']:.0f}s"
        f"（丟棄 {rec.clip_info['clipped_sec']:.0f}s）"
    )

    m = su.sleep_metrics(rec.stages, rec.epoch_sec)
    print(
        f"TIB={su.fmt_hms(m['tib_sec'])}  TST={su.fmt_hms(m['tst_sec'])}  "
        f"SE={m['sleep_efficiency'] * 100:.1f}%  SOL={su.fmt_hms(m['sol_sec'])}  "
        f"REM lat={su.fmt_hms(m['rem_latency_sec'])}  WASO={su.fmt_hms(m['waso_sec'])}  "
        f"覺醒 {m['n_awakenings']} 次  無效 epoch {m['n_invalid']} 個"
    )

    print("計算各階段 Welch PSD ...")
    psd_blocks = compute_stage_psd(rec)
    rel = band_power_table(psd_blocks)

    print("繪圖 ...")
    figs = {
        "hypnogram": img_tag(plot_hypnogram(rec), "Hypnogram"),
        "pie": img_tag(plot_stage_pie(m), "Stage distribution pie chart"),
        "psd_fpz": img_tag(
            plot_psd("EEG Fpz-Cz", psd_blocks["EEG Fpz-Cz"]), "PSD EEG Fpz-Cz"
        ),
        "psd_pz": img_tag(
            plot_psd("EEG Pz-Oz", psd_blocks["EEG Pz-Oz"]), "PSD EEG Pz-Oz"
        ),
        "bands": img_tag(plot_band_bars(rel), "Relative band power"),
    }

    OUT_HTML.write_text(build_html(rec, m, figs, rel), encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"完成 -> {OUT_HTML}（{size_kb:.0f} KB）")

    if size_kb < 200:
        print("警告：HTML 小於 200 KB，圖可能沒有內嵌成功")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
