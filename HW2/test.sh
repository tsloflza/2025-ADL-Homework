#!/bin/bash
set -e

# ⚠️ Modify these first
PPL_PATH="ppl.py"
MODEL_PATH="Qwen/Qwen3-4B"
PEFT_PATH="./adapter_checkpoint"
PUBLIC_FILE="public_test.json"
OUTPUT_FILE="./predictions.json"
# ⚠️ ppl.py utils.py test.sh run.sh download.sh 放在同一個目錄底下

# requirements
pip install pipreqs cryptography > /dev/null 2>&1
echo

echo "=============================="
echo " Step 1. Environment check"
echo "=============================="
echo

# 助教的環境 (硬體、OS)
echo "[TA Environment]"
echo "OS       : Ubuntu 20.04"
echo "Python   : 3.10"
echo "GPU      : RTX 3070 8GB VRAM"
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
echo "bitsandbytes==0.44.1"
echo "gdown"
echo "peft==0.13.0"
echo "torch==2.4.1"
echo "transformers>=4.51.0"
echo

# ========== My package list ==========
echo "[My package list]"

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

bash ./download.sh

end_time=$(date +%s)
echo
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

bash ./run.sh "$MODEL_PATH" "$PEFT_PATH" "$PUBLIC_FILE" "$OUTPUT_FILE"

end_time=$(date +%s)
echo "run.sh runtime: $((end_time - start_time)) sec"

# # 還原網路
# if command -v iptables >/dev/null 2>&1; then
#   echo "[INFO] Restoring internet..."
#   sudo iptables -D OUTPUT ! -o lo -p tcp --dport 22 -j DROP
# fi
echo

echo "=============================="
echo " Step 5. Test public perplexity"
echo "=============================="
echo

start_time=$(date +%s)

python3 "$PPL_PATH" \
    --base_model_path "$MODEL_PATH" \
    --peft_path "$PEFT_PATH" \
    --test_data_path "$PUBLIC_FILE"

end_time=$(date +%s)
echo
echo "ppl.py runtime: $((end_time - start_time)) sec"
echo

echo "=============================="
echo " All tests completed"
echo "=============================="
echo