# RUN SUMMARY — 睡眠報告 + YASA 自動分期比對

資料：`SC4002E0-PSG.edf` / `SC4002EC-Hypnogram.edf`

## 裁切驗證

| | epoch 範圍 | 長度 |
| --- | --- | --- |
| 裁切前 | 0–2829 | **23.58 小時** |
| 裁切後（TIB） | 809–1936 | **9.40 小時**（21:34 → 06:58）|

裁切確實生效。標註總跨度 86400 秒 > 訊號 84900 秒，超出的 1500 秒
（尾端 `Sleep stage ?`）整段 clip 掉。**被排除的無效 epoch：1 個**
（`Movement time`，30 秒），已從所有統計、accuracy、kappa 中排除。

## 練習 A — 七項指標

| 指標 | 數值 | 判讀 |
| --- | --- | --- |
| Sleep Efficiency | **83.7%** | 合理。略低於 85–95%，因為 WASO 偏高，但屬正常範圍邊緣。 |
| TST | 7h 52m | 合理。落在成人 7–9 小時正中間。 |
| SOL | 30m 00s | **異常（結構性）**。裁切規則規定往前保留 30 分鐘，SOL 必然恆等於 30 分，不是量到的值。真正的 SOL 需要 lights-off 標記，Sleep-EDF 沒有提供。 |
| REM Latency | 1h 06m | 合理。略短於 70–120 分，但同一夜有 5–6 個完整 REM 週期，型態正常。 |
| WASO | 1h 02m | 偏高。正常 <30 分。此人整夜片段化明顯（見覺醒次數）。 |
| 覺醒次數 | 21 次 | 偏多但合理，與 WASO 偏高一致。 |
| 各階段佔 TST | N1 6.3% / N2 39.5% / N3 31.5% / R 22.8% | 合理。REM 22.8% 正中參考值；N3 31.5% 偏高、N2 39.5% 偏低，是 Sleep-EDF 用 R&K 標準（stage 3+4 合併成 N3）的已知效應。 |

功率頻譜符合生理預期：N3 的 delta 佔比最高（Fpz-Cz 92%），
N2 在 13 Hz 有明顯紡錘波峰，W 在 Pz-Oz 有 28% alpha（後頭部閉眼 alpha）。

## 練習 B — YASA 比對

**accuracy 0.9219、Cohen's kappa 0.8521、macro F1 0.7608**（2829 個 epoch）

| 階段 | W | N1 | N2 | N3 | R |
| --- | --- | --- | --- | --- | --- |
| F1 | 0.981 | **0.333** | 0.787 | 0.936 | 0.767 |

**最差是 N1**（F1 0.333、recall 0.271，support 只有 59）。
從 confusion matrix 看，N1 有 **59.3% 被誤判成 N2**、5.1% 成 W。
次差是 REM（recall 0.642），**32.1% 被誤判成 N2**。
誤判幾乎全部流向 N2，與「只有 Fpz-Cz 額葉導程、沒有 YASA 訓練用的 C3/C4 中央導程」
一致：額葉的紡錘波振幅小、慢波相對強，N1/REM 的鑑別特徵被削弱。

只看裁切後 TIB 區間的話 accuracy 降到 0.8287、kappa 0.7685 —— 整段錄音的
0.92 有一部分是靠 1885 個 W epoch 撐起來的，這個數字更能代表真實難度。

## 遇到的問題與解法

1. **YASA 0.7 的 `predict()` 回傳 `yasa.Hypnogram` 而非字串陣列**，
   該物件沒有 `__iter__` 卻有 `__getitem__`，且任何索引都回傳新的 1-epoch
   Hypnogram、永不丟 `IndexError` → 直接 `for s in pred` 會**無限迴圈卡死**
   （第一次執行就是這樣掛掉的）。改走 `.hypno` 這個 pandas Series。
2. **同一個物件的標籤是 `WAKE`/`REM` 不是 `W`/`R`**，照舊 mapping 會全部變成 -1。
   兩種都收，並加一道 guard：出現無法對應的標籤就直接報錯，不靜默吞掉。
3. **`OUT_DIR.mkdir(exist_ok=True)` 在父目錄不存在時會炸**——
   由下面第 7 項的乾淨狀態測試抓到，改成 `parents=True`。
4. **sklearn 版本警告**：YASA 內建分類器是用 sklearn 0.24.2 pickle 的，
   本環境是 1.9.0，會跳 `InconsistentVersionWarning`。
   結果數值合理且可重現（跑兩次完全相同），判定為良性，未做處置。
5. 依 brief 只餵 EEG + EOG，不傳 `emg_name`；`.venv-yasa` 獨立建立，
   現有 `.venv` 全程未安裝任何套件。

## 自我驗證

| # | 項目 | 結果 |
| --- | --- | --- |
| 1 | `sleep_report.html` > 200 KB | ✅ **458 KB** |
| 2 | Sleep Efficiency 落在 60–98% | ✅ 83.7% |
| 3 | 裁切後 TIB 遠小於 23.58h（預期 8–11h） | ✅ 9.40h |
| 4 | TST + WASO + SOL ≈ TIB（誤差 < 1 分） | ✅ 563.5 vs 564.0 分，差 **30 秒**（正好是那 1 個無效 epoch；計入後誤差為 0） |
| 5 | `yasa_metrics.json` 存在、accuracy 在 0.5–0.95 | ✅ 0.9219 |
| 6 | Cohen's Kappa > 0.3 | ✅ 0.8521 |
| 7 | 兩支腳本能從乾淨狀態重跑並 exit 0 | ✅ 見下方說明 |

第 7 項說明：`report_sleep.py` 導向一個全新、不存在的輸出目錄重跑 → exit 0、
產出 458 KB（就是這一輪抓到第 3 個問題）。`yasa_compare.py` 完整重跑一次 →
exit 0，且 accuracy / kappa / 每階段 F1 與前一次**完全相同**。
原本打算先刪掉 `report/` 再重跑，但刪除指令被 shell hook 攔下
（環境也沒有 `trash`），依 SECURITY PREFIX「失敗不繞路」的規定沒有嘗試繞過，
改用上述「導向全新目錄」的方式驗證，效果等價。

沒有任何一項需要反覆修（唯一連續修的是第 7 項連帶抓到的 `mkdir`，一次修好）。
