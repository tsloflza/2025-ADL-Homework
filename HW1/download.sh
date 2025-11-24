#!/bin/bash
set -e

# === 安裝 gdown ===
pip install -q gdown

# === Google Drive 資料夾 ID ===
FOLDER_ID="1UhMSg6jF3zUt-vPKZd95YCOSl7du0pG4"

# === 輸出路徑 ===
OUT_DIR="./downloads"
mkdir -p $OUT_DIR

# === 下載整個資料夾 ===
echo "Downloading models data from Google Drive..."
gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID -O $OUT_DIR

echo "Download finished. Files saved to $OUT_DIR"

# === 解壓縮 ===
unzip $OUT_DIR/ntu-adl-2025-hw-1.zip -d $OUT_DIR
unzip $OUT_DIR/ps2_model.zip -d $OUT_DIR
unzip $OUT_DIR/ss2_model.zip -d $OUT_DIR
