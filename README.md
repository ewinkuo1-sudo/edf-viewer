# EDF Viewer

用 Streamlit + MNE + Plotly 瀏覽 EDF 訊號波形（含 Sleep-EDF Hypnogram 睡眠分期）。

## 安裝

這台機器上沒有 conda，改用專案內的 venv：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

（如果你之後裝了 conda，`conda activate bioagent` + `pip install -r requirements.txt` 也一樣。）

## 資料

把 `.edf` 檔放進 `data/`：

```
data/SC4002E0-PSG.edf
data/SC4002EC-Hypnogram.edf
```

Hypnogram 會依檔名前 6 碼（`SC4002`）自動配對，不用手動選。

## 啟動

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

瀏覽器會開 http://localhost:8501

## 功能

- 從 `data/` 選檔或直接上傳 `.edf`
- 頂部一行 caption：檔名 / 頻道數 / 取樣率 / 總時長 / 錄製時間
- Sidebar：頻道多選、epoch 長度
- 導覽列（波形圖正上方，同一行）：−100 / −1 / epoch 輸入框 / +1 / +100 / 跳到下一個 W、N1、N2、N3、REM
- 導覽列下方以對應顏色顯示目前 epoch 的睡眠階段
- Plotly 波形圖，可縮放平移（滾輪縮放已開啟），每個頻道各自一條 y 軸
- 波形圖下方 expander 收合各頻道統計（max / min / mean / std）
