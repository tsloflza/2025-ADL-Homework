#!/bin/bash
set -e

# === Google Drive 資料夾 ID ===
FOLDER_ID="1A6aixZriGfVJYqI7ba1fgE-pHKxtmOFU"

# === 輸出路徑 ===
OUT_DIR="./adapter_checkpoint"
mkdir -p $OUT_DIR

# === 下載整個資料夾 ===
echo "Downloading models data from Google Drive..."
gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID -O $OUT_DIR

echo "Download finished. Files saved to $OUT_DIR"
