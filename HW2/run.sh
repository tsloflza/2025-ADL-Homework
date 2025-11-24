
# 檢查參數數量
if [ "$#" -ne 4 ]; then
    echo "Usage: bash run.sh <model_path> <adapter_path> <input_file> <output_file>"
    exit 1
fi

MODEL=$1
ADAPTER=$2
INPUT=$3
OUTPUT=$4

# 執行 py
python inference.py \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --input_path "$INPUT" \
  --output_path "$OUTPUT" \
