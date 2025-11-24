#!/bin/bash
set -e

# ⚠️ Modify these first
CONTEXT_PATH=ntu-adl-2025-hw-1/context.json
TEST_PATH=ntu-adl-2025-hw-1/test.json
PREDICT_PATH=prediction.csv
# ⚠️ test.sh run.sh download.sh 放在同一個目錄底下

echo "=============================="
echo " Step 1. Environment check"
echo "=============================="
echo

# 助教的環境 (硬體、OS)
echo "[TA Environment]"
echo "OS       : Ubuntu 20.04"
echo "Python   : 3.10"
echo "GPU      : RTX 2080 Ti 11GB VRAM"
echo "Memory   : 32GB RAM"
echo "DiskFREE : 20GB"
echo

# 我的環境
echo "[Your Environment]"
# OS & Kernel
echo -n "OS       : "
uname -o

# Python
echo -n "Python   : "
python3 --version | awk '{print $2}'

# GPU (只印型號 + 顯存)
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)
    GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
    GPU_MEM=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
    echo "GPU      : $GPU_NAME ${GPU_MEM}MB"
else
    echo "GPU      : No GPU detected"
fi

# RAM
echo -n "Memory   : "
free -h | awk '/^Mem:/ {print $2}'

# Disk (目前目錄所在磁碟剩餘)
echo -n "DiskFree : "
df -h . | awk 'NR==2 {print $4 " free of " $2}'
echo

echo "=============================="
echo " Step 2. Package check"
echo "=============================="
echo

# ========== TA package list ==========
echo "[TA package list]"
echo "accelerate==0.34.2"
echo "datasets==2.21.0"
echo "evaluate"
echo "gdown"
echo "matplotlib"
echo "nltk==3.9.1"
echo "numpy"
echo "pandas"
echo "scikit-learn==1.5.1"
echo "torch==2.1.0"
echo "tqdm"
echo "transformers==4.50.0"
echo

# ========== My package list (pipreqs) ==========
echo "[My package list]"

# pipreqs 會產生 requirements.txt，版本號是直接抓最新的
pip install pipreqs >/dev/null 2>&1

# 存成暫時檔
TMP_REQS="my_requirements.txt"

# 強制覆蓋並掃描當前目錄
pipreqs . --force --savepath "$TMP_REQS" >/dev/null 2>&1

if [ -f "$TMP_REQS" ]; then
  cat "$TMP_REQS"
  rm "$TMP_REQS"
else
  echo "pipreqs failed to generate requirements"
fi
echo

echo "=============================="
echo " Step 3. Run download.sh"
echo "=============================="
echo

start_time=$(date +%s)

bash ./download.sh > download.log 2>&1

end_time=$(date +%s)
echo "download.sh runtime: $((end_time - start_time)) sec"
echo

echo "=============================="
echo " Step 4. Run run.sh"
echo "=============================="
echo

# 模擬斷網 (⚠️ sudo 權限)
# if command -v iptables >/dev/null 2>&1; then
#   echo "[INFO] Simulating no internet (blocking OUTPUT except SSH)..."
#   sudo iptables -I OUTPUT ! -o lo -p tcp --dport 22 -j DROP
# fi

start_time=$(date +%s)

bash ./run.sh "$CONTEXT_PATH" "$TEST_PATH" "$PREDICT_PATH" > run.log 2>&1

end_time=$(date +%s)
echo "run.sh runtime: $((end_time - start_time)) sec"

# # 還原網路
# if command -v iptables >/dev/null 2>&1; then
#   echo "[INFO] Restoring internet..."
#   sudo iptables -D OUTPUT ! -o lo -p tcp --dport 22 -j DROP
# fi
echo

echo "=============================="
echo " All tests completed"
echo "=============================="