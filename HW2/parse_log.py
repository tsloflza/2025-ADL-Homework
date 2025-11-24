import ast
import re
import os
from typing import List, Dict, Any
import csv

LOG_FILE_PATH = "outputs/old_qwen/tuning.log"
CSV_FILE_PATH = "training_metrics.csv"

# 用於儲存提取的結果
extracted_data: List[Dict] = []

def extract_training_metrics(log_file_path: str) -> List[Dict]:
    """
    從訓練日誌文件中提取 loss, eval_loss, 和 epoch，並將同一 epoch 的記錄合併。
    Args:
        log_file_path: 訓練日誌文件的路徑。
    Returns:
        包含提取指標的字典列表，其中 loss 和 eval_loss 在同一條記錄中。
    """
    
    if not os.path.exists(log_file_path):
        print(f"錯誤: 找不到日誌文件 '{log_file_path}'。")
        return []

    epoch_buffer: Dict[float, Dict[str, Any]] = {}
    
    # 用於匹配字典結構，並提取內容
    dict_pattern = re.compile(r'({.*?})')

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # 找到行中匹配的字典字符串
            match = dict_pattern.search(line.strip())

            if match:
                dict_str = match.group(1)
                try:
                    # 使用 ast.literal_eval 安全地將字符串轉換為字典
                    record = ast.literal_eval(dict_str)
                    
                    # 只有當 'epoch' 存在時才處理
                    if 'epoch' in record:
                        epoch_value = record['epoch']
                        
                        # 確保當前 epoch 記錄在 buffer 中已經初始化
                        if epoch_value not in epoch_buffer:
                            # 初始化時，將 epoch 本身作為記錄的一部分
                            epoch_buffer[epoch_value] = {'epoch': epoch_value}

                        # 檢查並合併所需的指標
                        if 'loss' in record:
                            epoch_buffer[epoch_value]['loss'] = record['loss']
                        
                        if 'eval_loss' in record:
                            epoch_buffer[epoch_value]['eval_loss'] = record['eval_loss']
                        
                except (ValueError, SyntaxError) as e:
                    # 如果解析失敗，跳過該行並發出警告
                    print(f"警告: 第 {line_num+1} 行解析字典失敗，已跳過。內容: {dict_str[:50]}...")
                    continue

    results = sorted(epoch_buffer.values(), key=lambda x: x['epoch'])
    return results

def save_to_csv(data: List[Dict], filename: str):
    """將提取的訓練指標數據保存到 CSV 文件中。"""
    
    # 合併欄位
    fieldnames = ['epoch', 'loss', 'eval_loss']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 寫入標題行
            writer.writeheader()
            
            # 寫入數據行
            for row in data:
                writer.writerow(row)
                
        print(f"✅ 數據已成功保存到 CSV 檔案: {filename}")
        
    except Exception as e:
        print(f"❌ 儲存 CSV 失敗: {e}")


extracted_data = extract_training_metrics(LOG_FILE_PATH)
save_to_csv(extracted_data, CSV_FILE_PATH)