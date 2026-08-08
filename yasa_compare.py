"""練習 B —— YASA 自動分期 vs. 人工標註。

跑法（用獨立的 .venv-yasa，不是 .venv）：

    .\\.venv-yasa\\Scripts\\python.exe yasa_compare.py

輸出：
    report/yasa_metrics.json
    report/yasa_hypnogram_compare.png
    report/yasa_confusion.png
    report/yasa_classification_report.png
    report/yasa_classification_report.txt

注意：只餵 EEG + EOG，不傳 emg_name。EMG submental 原始只有 1 Hz，
被 MNE 上採樣成階梯訊號，餵給 YASA 只會拉低準確率。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import yasa
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

import sleep_utils as su

mne.set_log_level("ERROR")

# --------------------------------------------------------------------------
# 受試者 metadata
# --------------------------------------------------------------------------
# 假設值。真值需查 Sleep-EDF 的 SC-subjects.xls，本專案沒有這個檔案。
# YASA 會把 age / male 當成特徵餵給分類器，填錯只會小幅影響結果，不會讓它壞掉。
AGE = 25
MALE = True

EEG_NAME = "EEG Fpz-Cz"
EOG_NAME = "EOG horizontal"

OUT_DIR = Path(__file__).parent / "report"

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8a85"
GRID = "#ececea"
DISAGREE = "#e34948"

# YASA 回傳的字串標籤 -> sleep_utils 的整數編碼。
# yasa.Hypnogram 用的是 WAKE / REM，分類器內部用的是 W / R，兩種都收。
YASA_TO_CODE = {
    "W": su.W,
    "WAKE": su.W,
    "N1": su.N1,
    "N2": su.N2,
    "N3": su.N3,
    "R": su.R,
    "REM": su.R,
}


def prediction_labels(pred_obj) -> np.ndarray:
    """把 predict() 的回傳值取成字串標籤陣列。

    YASA 0.7 的 predict() 回傳的是 `yasa.Hypnogram`，不是字串陣列。
    這個物件沒有 `__iter__`，卻有 `__getitem__` —— 而且任何索引都回傳一個新的
    1-epoch Hypnogram、永遠不丟 IndexError，所以直接 `for s in pred` 會無限迴圈
    （不是報錯，是整支程式卡死）。一律走 `.hypno` 這個 pandas Series。
    """
    hypno = getattr(pred_obj, "hypno", pred_obj)
    return np.asarray(hypno, dtype=str)

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
# YASA 自動分期
# --------------------------------------------------------------------------
def run_yasa(psg_path: Path) -> tuple[np.ndarray, np.ndarray, object]:
    """回傳 (預測階段碼, 每個 epoch 的信心值, raw)。"""
    print(f"讀取 {psg_path.name}（只載入 {EEG_NAME} + {EOG_NAME}）...")
    raw = mne.io.read_raw_edf(str(psg_path), preload=False)
    raw.pick([EEG_NAME, EOG_NAME])
    raw.load_data()

    print(f"YASA 分期中（age={AGE}, male={MALE}，不傳 emg_name）...")
    sls = yasa.SleepStaging(
        raw,
        eeg_name=EEG_NAME,
        eog_name=EOG_NAME,
        metadata=dict(age=AGE, male=MALE),
    )
    sls.fit()
    labels = prediction_labels(sls.predict())
    proba = sls.predict_proba()

    pred = np.array([YASA_TO_CODE.get(s, su.INVALID) for s in labels], dtype=int)
    if (pred == su.INVALID).any():
        bad = sorted(set(labels[pred == su.INVALID]))
        raise RuntimeError(f"YASA 回傳無法對應的階段標籤：{bad}")
    conf = np.asarray(proba.max(axis=1), dtype=float)
    print(f"YASA 產出 {len(pred)} 個 epoch，平均信心 {conf.mean():.3f}")
    return pred, conf, raw


# --------------------------------------------------------------------------
# 對齊 + 評分
# --------------------------------------------------------------------------
def align(human: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> dict:
    """兩邊都從錄音起點切 30 秒 epoch，取共同長度，評分時排除人工標註 -1。"""
    n = min(len(human), len(pred), len(conf))
    human, pred, conf = human[:n], pred[:n], conf[:n]
    valid = human != su.INVALID
    return {
        "n_common": n,
        "n_human": len(human),
        "n_yasa": len(pred),
        "human": human,
        "pred": pred,
        "conf": conf,
        "valid": valid,
        "n_excluded": int((~valid).sum()),
    }


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(su.STAGE_CODES)
    names = list(su.STAGE_LABELS)
    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "n_epochs_scored": int(len(y_true)),
        "per_stage": {
            name: {
                "precision": float(rep[name]["precision"]),
                "recall": float(rep[name]["recall"]),
                "f1": float(rep[name]["f1-score"]),
                "support": int(rep[name]["support"]),
            }
            for name in names
        },
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_labels": names,
    }


# --------------------------------------------------------------------------
# 圖 1 — 人工 vs YASA hypnogram
# --------------------------------------------------------------------------
def plot_compare(rec: su.Recording, a: dict, out: Path) -> None:
    n = a["n_common"]
    offsets = np.arange(n + 1) * rec.epoch_sec
    t = mdates.date2num(su.clock_datetimes(rec.meas_date, offsets))

    ypos = {code: i for i, code in enumerate(su.PLOT_ORDER)}

    def to_y(stages):
        y = np.array([ypos.get(int(s), np.nan) for s in stages], dtype=float)
        return np.append(y, y[-1])

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[3, 0.42, 3, 2], hspace=0.16
    )
    ax_h = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1], sharex=ax_h)
    ax_y = fig.add_subplot(gs[2], sharex=ax_h)
    ax_c = fig.add_subplot(gs[3], sharex=ax_h)

    def draw_hypno(ax, stages, title, color):
        ax.step(t, to_y(stages), where="post", color=color, lw=0.9)
        for stage, i, j in su.stage_runs(stages):
            if stage == su.R:
                ax.step(
                    t[i : j + 1],
                    np.full(j - i + 1, float(ypos[su.R])),
                    where="post",
                    color=su.STAGE_COLORS[su.R],
                    lw=2.8,
                    solid_capstyle="butt",
                )
            elif stage == su.INVALID:
                ax.axvspan(t[i], t[j], color=su.STAGE_COLORS[su.INVALID], alpha=0.5, lw=0)
        ax.set_yticks(range(len(su.PLOT_ORDER)))
        ax.set_yticklabels([su.STAGE_LABELS[c] for c in su.PLOT_ORDER])
        ax.set_ylim(len(su.PLOT_ORDER) - 0.5, -0.5)
        ax.set_ylabel(title, fontsize=10.5, labelpad=8)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    draw_hypno(ax_h, a["human"], "Manual scoring", "#3a3a38")
    draw_hypno(ax_y, a["pred"], "YASA", "#1d5297")

    # 中間：不一致的 epoch 用紅色細條標出（只看人工標註有效的部分）
    mismatch = a["valid"] & (a["human"] != a["pred"])
    idx = np.flatnonzero(mismatch)
    ax_d.vlines(t[idx], 0, 1, color=DISAGREE, lw=0.55, alpha=0.85)
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([])
    ax_d.set_ylabel(
        f"Disagree\n{mismatch.sum()}/{int(a['valid'].sum())}",
        fontsize=8.5,
        color=DISAGREE,
        rotation=0,
        ha="right",
        va="center",
        labelpad=12,
    )
    for side in ("top", "right", "left", "bottom"):
        ax_d.spines[side].set_visible(False)
    # 不關掉 tick marks 的話，共用 x 軸的刻度會在紅條下方留下一排黑點，
    # 看起來像是額外的不一致標記
    ax_d.tick_params(length=0)

    # 信心曲線
    ax_c.fill_between(
        t[:-1], a["conf"], 0.2, color="#2a78d6", alpha=0.22, lw=0, step="post"
    )
    ax_c.step(t[:-1], a["conf"], where="post", color="#1d5297", lw=0.8)
    ax_c.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax_c.text(
        t[0], 0.505, " confidence = 0.5", fontsize=8, color=MUTED, va="bottom"
    )
    ax_c.set_ylim(0.2, 1.02)
    ax_c.set_ylabel("YASA\nconfidence", fontsize=10.5, labelpad=8)
    ax_c.grid(axis="y", color=GRID, lw=0.7)
    ax_c.set_axisbelow(True)
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    for ax in (ax_h, ax_d, ax_y):
        ax.tick_params(labelbottom=False)
    ax_c.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_c.set_xlabel("Clock time (HH:MM)")
    ax_h.set_xlim(t[0], t[-1])

    # 標出裁切後的 TIB 區間，方便對照練習 A
    tib0, tib1 = (
        mdates.date2num(
            su.clock_datetimes(
                rec.meas_date,
                np.array(
                    [rec.trim_start * rec.epoch_sec, (rec.trim_stop + 1) * rec.epoch_sec]
                ),
            )
        )
    )
    for ax in (ax_h, ax_y, ax_c):
        for x in (tib0, tib1):
            ax.axvline(x, color="#c98500", lw=1.0, ls=":", zorder=5)

    ax_h.set_title(
        "Manual scoring vs. YASA automatic staging "
        f"(whole recording, {n} epochs; dotted = trimmed TIB window)",
        fontsize=12.5,
        pad=12,
    )
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# --------------------------------------------------------------------------
# 圖 2 — Confusion matrix
# --------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, names: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws=dict(label="Epoch count"),
        xticklabels=names,
        yticklabels=names,
        linewidths=0.6,
        linecolor="white",
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix — raw counts", fontsize=12, pad=10)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws=dict(label="Share of manual-scored epochs"),
        xticklabels=names,
        yticklabels=names,
        linewidths=0.6,
        linecolor="white",
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix — row-normalised (each row sums to 1)",
                      fontsize=12, pad=10)

    for ax in axes:
        ax.set_xlabel("YASA prediction")
        ax.set_ylabel("Manual scoring (ground truth)")
        ax.tick_params(rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# --------------------------------------------------------------------------
# 圖 3 — classification report
# --------------------------------------------------------------------------
def plot_classification_report(s: dict, out: Path) -> None:
    names = list(su.STAGE_LABELS)
    metrics = ["precision", "recall", "f1"]
    mat = np.array([[s["per_stage"][n][m] for m in metrics] for n in names])
    support = [s["per_stage"][n]["support"] for n in names]

    fig, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1])
    )

    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=[m.capitalize() for m in metrics],
        yticklabels=[f"{n}  (n={sup})" for n, sup in zip(names, support)],
        linewidths=0.6,
        linecolor="white",
        cbar_kws=dict(label="Score"),
        ax=axes[0],
    )
    axes[0].set_title(
        f"Per-stage performance   —   accuracy {s['accuracy']:.3f},"
        f"  kappa {s['cohen_kappa']:.3f}",
        fontsize=12,
        pad=10,
    )
    axes[0].tick_params(rotation=0)

    x = np.arange(len(names))
    width = 0.26
    colors = ["#5598e7", "#2a78d6", "#1c5cab"]
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = mat[:, i]
        axes[1].bar(x + (i - 1) * width, vals, width, label=m.capitalize(), color=c)
        for xi, v in zip(x + (i - 1) * width, vals):
            axes[1].text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5,
                         color=SECONDARY)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 1.12)
    axes[1].set_xlabel("Sleep stage")
    axes[1].set_ylabel("Score")
    # macro F1 直接掛在 legend 上，不用 text 標在線上 —— 標在線上會壓到 R 的長條標籤
    axes[1].axhline(
        s["macro_f1"],
        color="#c98500",
        lw=1.1,
        ls="--",
        label=f"macro F1 = {s['macro_f1']:.3f}",
    )
    axes[1].legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    axes[1].set_title("Precision / recall / F1 by stage", fontsize=12, pad=10)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.name}")


# --------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rec = su.load_recording(preload=False)
    print(rec.describe_trim())

    pred, conf, _raw = run_yasa(rec.psg_path)
    a = align(rec.stages_full, pred, conf)
    print(
        f"對齊：人工 {a['n_human']} / YASA {a['n_yasa']} -> 共同 {a['n_common']} 個 epoch；"
        f"排除人工標註無效的 {a['n_excluded']} 個"
    )

    y_true, y_pred = a["human"][a["valid"]], a["pred"][a["valid"]]
    s = score(y_true, y_pred)
    print(f"\naccuracy      = {s['accuracy']:.4f}")
    print(f"Cohen's kappa = {s['cohen_kappa']:.4f}")
    print(f"macro F1      = {s['macro_f1']:.4f}")
    for name in su.STAGE_LABELS:
        p = s["per_stage"][name]
        print(
            f"  {name:<3} precision {p['precision']:.3f}  recall {p['recall']:.3f}  "
            f"F1 {p['f1']:.3f}  support {p['support']}"
        )

    # 次要分析：只看裁切後的 TIB 區間（練習 A 的分析範圍）
    tib_slice = slice(rec.trim_start, min(rec.trim_stop + 1, a["n_common"]))
    tib_valid = a["valid"][tib_slice]
    s_tib = score(a["human"][tib_slice][tib_valid], a["pred"][tib_slice][tib_valid])
    print(
        f"\n（參考）只看裁切後 TIB 區間：accuracy {s_tib['accuracy']:.4f}，"
        f"kappa {s_tib['cohen_kappa']:.4f}"
    )

    text = classification_report(
        y_true,
        y_pred,
        labels=list(su.STAGE_CODES),
        target_names=list(su.STAGE_LABELS),
        digits=3,
        zero_division=0,
    )
    report_txt = (
        f"YASA vs manual scoring — {rec.psg_path.name}\n"
        f"eeg_name={EEG_NAME}, eog_name={EOG_NAME}, emg_name=None\n"
        f"metadata: age={AGE}, male={MALE}  (假設值，真值需查 SC-subjects.xls)\n"
        f"epochs: common={a['n_common']}, scored={s['n_epochs_scored']}, "
        f"excluded (manual == -1)={a['n_excluded']}\n"
        f"accuracy={s['accuracy']:.4f}  cohen_kappa={s['cohen_kappa']:.4f}  "
        f"macro_f1={s['macro_f1']:.4f}\n\n{text}\n"
    )
    (OUT_DIR / "yasa_classification_report.txt").write_text(report_txt, encoding="utf-8")

    print("\n繪圖 ...")
    plot_compare(rec, a, OUT_DIR / "yasa_hypnogram_compare.png")
    plot_confusion(
        np.array(s["confusion_matrix"]),
        s["confusion_matrix_labels"],
        OUT_DIR / "yasa_confusion.png",
    )
    plot_classification_report(s, OUT_DIR / "yasa_classification_report.png")

    metrics = {
        "recording": rec.psg_path.name,
        "hypnogram": rec.hypno_path.name,
        "yasa_version": yasa.__version__,
        "channels": {"eeg_name": EEG_NAME, "eog_name": EOG_NAME, "emg_name": None},
        "metadata": {
            "age": AGE,
            "male": MALE,
            "note": "假設值；真值需查 Sleep-EDF 的 SC-subjects.xls",
        },
        "epochs": {
            "epoch_sec": rec.epoch_sec,
            "manual": a["n_human"],
            "yasa": a["n_yasa"],
            "common": a["n_common"],
            "excluded_invalid": a["n_excluded"],
            "scored": s["n_epochs_scored"],
        },
        "confidence": {
            "mean": float(a["conf"].mean()),
            "median": float(np.median(a["conf"])),
            "min": float(a["conf"].min()),
            "frac_below_0.5": float((a["conf"] < 0.5).mean()),
        },
        # 主要結果：依 brief 規定，從錄音起點對齊整段錄音
        "accuracy": s["accuracy"],
        "cohen_kappa": s["cohen_kappa"],
        "macro_f1": s["macro_f1"],
        "weighted_f1": s["weighted_f1"],
        "per_stage": s["per_stage"],
        "confusion_matrix": s["confusion_matrix"],
        "confusion_matrix_labels": s["confusion_matrix_labels"],
        "full_recording": {
            "accuracy": s["accuracy"],
            "cohen_kappa": s["cohen_kappa"],
            "macro_f1": s["macro_f1"],
        },
        # 次要參考：只看練習 A 的 TIB 區間
        "trimmed_tib_window": {
            "trim_start_epoch": rec.trim_start,
            "trim_stop_epoch": rec.trim_stop,
            "accuracy": s_tib["accuracy"],
            "cohen_kappa": s_tib["cohen_kappa"],
            "macro_f1": s_tib["macro_f1"],
            "per_stage": s_tib["per_stage"],
            "confusion_matrix": s_tib["confusion_matrix"],
        },
    }
    (OUT_DIR / "yasa_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  -> yasa_metrics.json")

    ok = 0.5 <= s["accuracy"] <= 0.95 and s["cohen_kappa"] > 0.3
    print(
        f"\n自我驗證：accuracy {s['accuracy']:.3f} 在 0.5–0.95 且 "
        f"kappa {s['cohen_kappa']:.3f} > 0.3 -> {'通過' if ok else '未通過'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
