# Brief: EDF Viewer 後續 — 睡眠報告 + YASA 自動分期比對

工作目錄：`C:\Users\ewink\edf-viewer`（Git Bash 下是 `~/edf-viewer`）

這是課程作業的兩個練習（A 睡眠報告 / B YASA 自動分期），兩個都要完成。
資料已在 `data/`，不需要下載。

---

## 已探勘的事實（不要重新猜，直接照用）

用 `.venv/Scripts/python.exe` + mne 讀過 header，確認：

- **PSG 頻道名稱**：`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, `Resp oro-nasal`,
  `EMG submental`, `Temp rectal`, `Event marker`
- **MNE 回報 sfreq = 100 Hz**，但 Resp / EMG / Temp 原始只有 1 Hz，
  是被 MNE 上採樣成階梯訊號 → **這三個頻道不可拿來做頻譜或分期**
- **錄音長度 23.58 小時**
- **Hypnogram 標註分佈**：W×23, 1×32, 2×37, 3×30, 4×16, R×12,
  `Movement time`×1, `Sleep stage ?`×1
- **第一個標註是 `Sleep stage W`, onset=0.0, duration=26070 秒（7.24 小時）**
  → 睡前躺著清醒佔了 7 小時多
- 標註總跨度 84900+1500 = 86400 秒（24h），**比實際資料還長**

### 三個必踩的坑（沒處理的話數字全錯）

1. **不裁切 → 睡眠效率算出來會是 ~25%**（正常 85–95%）。必須裁掉睡前/睡後的長 W。
2. **餵 EMG 給 YASA → 階梯假訊號拉低準確率**。只餵 EEG + EOG。
3. **裝 yasa 可能弄壞現有 .venv**（現況 numpy 2.5 / pandas 3.0 偏新，
   yasa 依賴鏈偏舊）→ 練習 B 開獨立 `.venv-yasa`，**不要動現有 .venv**。

---

## 任務

### 共用前置處理（A 和 B 都要用同一套，抽成 `sleep_utils.py`）

- 讀 hypnogram annotations，展開成 30 秒 epoch 的階段陣列。
- 階段對應 AASM 五類：`W` / `N1`(stage 1) / `N2`(stage 2) / `N3`(stage 3 **和** 4 合併) / `R`。
- `Movement time` 和 `Sleep stage ?` → 標成 `-1`（無效），
  **從所有統計、準確率、kappa 中排除**，但要回報排除了幾個 epoch。
- 標註超出實際資料長度的部分要 clip 掉。
- **裁切規則**：找出第一個和最後一個非 W 的有效 epoch，
  往前/往後各保留 30 分鐘，這個區間即為 TIB（Time in Bed）。
  裁切前後的 epoch 索引都要印出來，方便驗證。

### 練習 A — `report_sleep.py`

用現有的 `.venv`（`.venv/Scripts/python.exe`），**不要在這支裝任何套件**。
產出單一自包含 HTML 到 `report/sleep_report.html`。

內容：

1. **Hypnogram 圖** — x 軸用時鐘時間 `HH:MM`（從 EDF meas_date 起算），
   y 軸階段由上而下 `W → R → N1 → N2 → N3`，階梯狀線圖，
   REM 區段用不同顏色標示；無效 epoch 用灰色標出。
2. **各階段時間與佔比** — 圓餅圖 + 表格。
   表格要有兩欄佔比：**佔 TIB** 和 **佔 TST**（TST = 總睡眠時間，不含 W）。
3. **Sleep Efficiency** = TST / TIB
4. **Sleep Onset Latency (SOL)** = 裁切後起點到第一個非 W epoch
5. **REM Latency** = 入睡點到第一個 R epoch
6. **WASO**（入睡後清醒總時數）+ **覺醒次數**
   （連續 W 區段數，扣掉睡前/睡後那兩段；只計 ≥30 秒的區段）
7. **各階段 EEG 功率頻譜** — Welch PSD，
   `EEG Fpz-Cz` 和 `EEG Pz-Oz` 各一張，log-log 座標，
   每張圖上五條線（W/N1/N2/N3/R），
   並標出 delta 0.5–4 / theta 4–8 / alpha 8–12 / sigma 12–16 / beta 16–30 五個頻帶區塊。
   額外附一張各階段的相對頻帶功率長條圖（stacked bar）。

技術要求：

- **matplotlib 在 Windows 畫中文會變豆腐框** → 圖表標籤全部用英文，
  中文只出現在 HTML 的文字說明裡。
- 圖存 PNG 後 base64 內嵌進 HTML，**最終只有一個 .html 檔就能繳交**。
- 每個指標在報告裡附一句白話說明 + 健康成人正常範圍參考值。
- HTML 樣式要乾淨好看（白底、卡片式指標區塊、表格有斑馬紋），不要用外部 CDN。

### 練習 B — `yasa_compare.py`

1. 建獨立虛擬環境 `.venv-yasa`（`python -m venv .venv-yasa`），
   **允許在這個環境裡** `pip install` 以下套件（這是 SECURITY PREFIX 規則 4 的明確例外）：
   `yasa mne scikit-learn matplotlib seaborn pandas numpy scipy`
   —— 只限這些，不要裝別的。若安裝失敗，記錄完整錯誤並嘗試放寬版本一次；
   再失敗就停下來回報，**不要動現有 `.venv`**。
2. `yasa.SleepStaging(raw, eeg_name='EEG Fpz-Cz', eog_name='EOG horizontal')`
   —— **不要傳 `emg_name`**（理由見上面坑 2）。
3. `metadata` 的 `age` / `male` 做成腳本開頭的常數，
   預設 `AGE = 25, MALE = True`，並在 README 明確註記
   「真值需查 Sleep-EDF 的 SC-subjects.xls，目前為假設值」。
4. **對齊**：兩邊都從錄音起點切 30 秒 epoch，取共同長度；
   評分時只保留人工標註 != -1 的 epoch。
5. **對比圖**：上下兩排 hypnogram（上=人工，下=YASA），共用 x 軸；
   第三排畫 YASA 的 confidence（`predict_proba().max(axis=1)`）曲線；
   不一致的 epoch 在中間用紅色細條標出。
6. `accuracy` 和 `Cohen's Kappa`，印出來並存成 `report/yasa_metrics.json`。
7. **Confusion matrix 熱圖**，出兩版：原始次數 + row-normalized（每列加總為 1）。
8. `sklearn.metrics.classification_report`：每階段 precision / recall / F1 / support，
   同時存成文字和圖表。
9. 圖全部存到 `report/`（`yasa_hypnogram_compare.png`、`yasa_confusion.png`、
   `yasa_classification_report.png`）。

---

## 輸出

- `sleep_utils.py`、`report_sleep.py`、`yasa_compare.py`
- `report/sleep_report.html`（自包含，可直接繳交）
- `report/yasa_metrics.json`
- `report/yasa_hypnogram_compare.png`、`report/yasa_confusion.png`、
  `report/yasa_classification_report.png`
- `requirements-yasa.txt`（`.venv-yasa` 的 `pip freeze`）
- 更新 `README.md`：加「進階分析」章節，說明兩支腳本怎麼跑
  （含各自要用哪個 venv 的完整指令），並加一段說明
  **YASA 是用 C3/C4 中央導程訓練的，這筆資料只有 Fpz-Cz 額葉導程，
  準確率會低於論文宣稱值，這是預期內的**。
- 更新 `.gitignore`：加入 `.venv-yasa/`、`__pycache__/`、`*.spawn.log`、
  `*.effective_brief.md`（PNG 和 HTML 要進版控，不要 ignore）

## 報告

跑完寫 `report/RUN_SUMMARY.md`（≤600 字）：

- 練習 A 的七項指標實際數值，各標一句「合理 / 異常，理由是…」
- 裁切前後的 TIB 對照（裁切前 23.58h → 裁切後 ?h），證明裁切有生效
- 被排除的無效 epoch 數
- 練習 B 的 accuracy、Cohen's Kappa、各階段 F1
- 哪個階段 YASA 表現最差、從 confusion matrix 看它最常誤判成什麼
- 遇到的問題和怎麼解的

## 自我驗證（跑完必須逐項確認，寫進 RUN_SUMMARY.md）

1. `report/sleep_report.html` 存在且 > 200 KB（代表圖有內嵌成功）
2. Sleep Efficiency 落在 **60–98%** 之間 → 不在範圍代表裁切沒做對，要回頭修
3. 裁切後 TIB 應該遠小於 23.58 小時（預期 8–11 小時）
4. TST + WASO + SOL 三者相加應該約等於 TIB（誤差 < 1 分鐘）
5. `report/yasa_metrics.json` 存在，`accuracy` 落在 **0.5–0.95**
   （太低代表對齊錯了，1.0 代表比對到自己）
6. Cohen's Kappa 應該 > 0.3
7. 兩支腳本都能從乾淨狀態重跑並 exit 0

任一項不過 → 修到過為止；**同一項連續修兩次還不過就停下來，在 RUN_SUMMARY.md
寫清楚卡在哪、你試過什麼，不要繼續盲改**。

## 不做

- 不要動 `data/` 裡的 EDF 原始檔
- 不要改 `app.py`（現有的 Streamlit viewer 要維持能跑）
- 不要在現有 `.venv` 裡 `pip install` 任何東西
- 不要 `git commit`、`git push`、`git reset`、`git checkout --`（版控由智翔本人處理）
- 不要下載任何額外資料集
- 不要送 Discord（通知由 wrapper 處理）
- 不要 spawn 子 agent 做 brief 範圍外的事
