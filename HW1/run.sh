
# 檢查參數數量
if [ "$#" -ne 3 ]; then
    echo "Usage: bash run.sh <context.json> <test.json> <prediction.csv>"
    exit 1
fi

CONTEXT=$1
TEST=$2
OUTPUT=$3

# 執行 py
python ps_inference.py --context_file "$CONTEXT" --test_file "$TEST"
python ss_inference.py --test_file "$TEST" --output_file "$OUTPUT"
