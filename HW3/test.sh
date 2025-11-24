#!/bin/bash
set -e

# ⚠️ Modify these first
save_embeddings_py_path="../save_embeddings.py"
inference_batch_py_path="../inference_batch.py"
DATA_DIR="../data"
QRELS_PATH="../data/qrels.txt"
TEST_DATA_PATH="../data/test_open.txt"
# ⚠️ test.sh 放在 學號 目錄下，並在該目錄下執行

# requirements
pip install pipreqs > /dev/null 2>&1
echo

echo "=============================="
echo " Step 1. Environment check"
echo "=============================="
echo

# 助教的環境 (硬體、OS)
echo "[TA Environment]"
echo "OS       : Ubuntu 20.04"
echo "Python   : 3.12"
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
    echo "GPU      : $GPU_NAME ${GPU_MEM}MB VRAM"
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
echo "accelerate==1.10.1"
echo "datasets==4.0.0"
echo "faiss-gpu-cu12==1.12.0"
echo "gdown"
echo "python-dotenv==1.1.1"
echo "sentence-transformers==5.1.0"
echo "torch==2.8.0"
echo "tqdm==4.67.1"
echo "transformers==4.56.1"
echo

# ========== My package list ==========
echo "[My package list]"

# 存成暫時檔
TMP_REQS="my_requirements.txt"

# 強制覆蓋並掃描當前目錄，忽略 code/ script/
pipreqs . --force --savepath "$TMP_REQS" --ignore "code,script" >/dev/null 2>&1

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
echo "download.sh runtime: $((end_time - start_time)) sec (< 1 hour)"
echo

# === 複製檔案 ===
echo
cp "$save_embeddings_py_path" . 2>/dev/null && echo "Copied save_embedding.py" || echo "Failed to copy save_embedding.py"
cp "$inference_batch_py_path" . 2>/dev/null && echo "Copied inference_batch.py" || echo "Failed to copy inference_batch.py"
echo

echo "=============================="
echo " Step 4. Run save_embeddings.py"
echo "=============================="
echo

start_time=$(date +%s)

python save_embeddings.py --retriever_model_path ./models/retriever --data_folder "$DATA_DIR" --build_db

end_time=$(date +%s)

echo
echo "save_embeddings.py runtime: $((end_time - start_time)) sec"
echo

echo "=============================="
echo " Step 5. Run inference_batch.py"
echo "=============================="
echo

start_time=$(date +%s)

python inference_batch.py \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker \
    --test_data_path "$TEST_DATA_PATH" \
    --qrels_path "$QRELS_PATH"

end_time=$(date +%s)

echo
echo "Baseline:"
echo "    recall@10: 0.780 ↑"
echo "    mrr@10: 0.695 ↑"
echo "    Bi-Encoder_CosSim: 0.340 ↑"

echo
echo "save_embeddings.py runtime: $((end_time - start_time)) sec"
echo

echo "=============================="
echo " Step 6. Check Directory"
echo "=============================="
echo

rm -rf models vector_database results
rm -f save_embeddings.py inference_batch.py

echo "Current directory structure:"
tree -L 2 || ls -R
echo

required_files=("download.sh" "utils.py" "report.pdf" "README.md")
optional_dirs=("code" "script")

# === 檢查檔案 ===
for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "怎麼沒有 $f"
    fi
done

# === 檢查多餘檔案/資料夾 ===
allowed=("${required_files[@]}" "${optional_dirs[@]}")

for item in *; do
    skip=false
    for a in "${allowed[@]}"; do
        if [ "$item" = "$a" ]; then
            skip=true
            break
        fi
    done
    if [ "$skip" = false ]; then
        echo "$item 不能繳交"
    fi
done
echo

echo "=============================="
echo " All tests completed"
echo "=============================="
echo