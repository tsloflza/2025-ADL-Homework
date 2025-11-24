import json
import logging
import os
import random
import sys
from typing import Dict, List, Set

from sentence_transformers import (
    SentenceTransformer,
    losses,
    InputExample,
    LoggingHandler,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from torch.utils.data import DataLoader


# 參數設定
class Args:
    corpus_path = "data/corpus.txt"
    train_path = "data/train.txt"
    qrels_path = "data/qrels.txt"
    output_dir = "./output/retriever"
    batch_size = 32
    num_epochs = 1
    evaluation_steps = 100
    seed = 42
args = Args()


# 設定日誌
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def load_jsonl_data(filepath: str) -> List[Dict]:
    """讀取 JSONL 檔案 (例如 corpus.txt, train.txt)"""
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        logger.error(f"錯誤：找不到檔案 {filepath}。")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"錯誤：解析 JSON 失敗 {filepath}。請檢查格式。")
        sys.exit(1)
    return data


def load_qrels_data(filepath: str) -> Dict[str, Dict[str, int]]:
    """讀取 qrels.txt 檔案 (JSON 格式)"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"錯誤：找不到檔案 {filepath}。")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"錯誤：解析 JSON 失敗 {filepath}。請檢查格式。")
        sys.exit(1)


def create_mnr_training_examples(
    queries: Dict[str, str],
    corpus: Dict[str, str],
    relevant_docs: Dict[str, Set[str]],
) -> List[InputExample]:
    """
    為 MultipleNegativesRankingLoss 建立訓練樣本：
    每個 InputExample 只包含 (query, positive_passage)。
    MNR loss 會自動使用 batch 內其他 positive 當作 in-batch negative。
    """
    train_samples = []
    if not corpus:
        logger.warning("語料庫為空，無法建立訓練樣本。")
        return train_samples

    logger.info("開始建立 MNR 訓練樣本...")
    for qid, pos_pids in relevant_docs.items():
        query_text = queries.get(qid)
        if not query_text:
            logger.warning(f"找不到 qid {qid} 對應的查詢文本，跳過。")
            continue

        query_text_prefixed = f"query: {query_text}"

        for pos_pid in pos_pids:
            pos_text = corpus.get(pos_pid)
            if not pos_text:
                logger.warning(f"找不到 pid {pos_pid} (qid: {qid}) 對應的段落文本，跳過。")
                continue

            pos_text_prefixed = f"passage: {pos_text}"

            train_samples.append(InputExample(texts=[query_text_prefixed, pos_text_prefixed]))

    logger.info(f"成功建立 {len(train_samples)} 個 MNR 訓練樣本。")
    return train_samples


if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-small"
    logger.info(f"載入預訓練模型：{model_name}")
    model = SentenceTransformer(model_name)

    # 1. 載入資料
    logger.info("載入資料...")
    corpus_data = load_jsonl_data(args.corpus_path)
    train_data = load_jsonl_data(args.train_path)
    qrels_data = load_qrels_data(args.qrels_path)

    # 2. 資料前處理
    # 語料庫：{pid: text}
    corpus = {p["id"]: p["text"] for p in corpus_data if "id" in p and "text" in p}
    # 查詢：{qid: rewrite}
    queries = {q["qid"]: q["rewrite"] for q in train_data if "qid" in q and "rewrite" in q}
    # 相關文件：{qid: set(pid)}
    relevant_docs = {
        qid: set(pids.keys())
        for qid, pids in qrels_data.items()
        if qid in queries and pids
    }
    logger.info(f"載入 {len(corpus)} 個段落，{len(queries)} 個查詢，{len(relevant_docs)} 個相關文件對映。")

    # 3. 切分訓練/驗證集 (基於 qid)
    all_qids = list(relevant_docs.keys())
    random.seed(args.seed)
    random.shuffle(all_qids)

    split_idx = int(len(all_qids) * 0.9)
    train_qids = all_qids[:split_idx]
    eval_qids = all_qids[split_idx:]

    logger.info(f"總 QIDs: {len(all_qids)} | 訓練 QIDs: {len(train_qids)} | 驗證 QIDs: {len(eval_qids)}")

    # 4. 建立訓練樣本 (MultipleNegativesRankingLoss 需要 (query, positive) pairs)
    train_relevant_docs = {qid: relevant_docs[qid] for qid in train_qids}
    train_samples = create_mnr_training_examples(queries, corpus, train_relevant_docs)
    if not train_samples:
        logger.error("沒有可用的訓練樣本，請檢查資料。")
        sys.exit(1)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # 5. 建立 Loss -> MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 6. 建立評估器 (InformationRetrievalEvaluator)
    eval_queries_ir = {qid: f"query: {queries[qid]}" for qid in eval_qids if qid in queries}
    eval_corpus_ir = {pid: f"passage: {text}" for pid, text in corpus.items()}
    eval_relevant_docs_ir = {qid: relevant_docs[qid] for qid in eval_qids if qid in relevant_docs and relevant_docs[qid]}

    if eval_queries_ir and eval_corpus_ir and eval_relevant_docs_ir:
        evaluator_ir = InformationRetrievalEvaluator(
            queries=eval_queries_ir,
            corpus=eval_corpus_ir,
            relevant_docs=eval_relevant_docs_ir,
            name="eval_recall",
            main_score_function="cosine",
            show_progress_bar=False,
            mrr_at_k=[1, 3, 5, 10],
            precision_recall_at_k=[1, 3, 5, 10],
            batch_size=32,
        )
        logger.info(f"建立 'InformationRetrievalEvaluator'")
        evaluators = [evaluator_ir]
        evaluator = SequentialEvaluator(evaluators)
    else:
        logger.warning("驗證資料不足，無法建立 InformationRetrievalEvaluator。")
        evaluator = None

    # 7. 訓練模型
    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1)  # 10% 的 warmup

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("開始模型訓練...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        evaluation_steps=args.evaluation_steps,
        save_best_model=True,  # 儲存最佳模型
        use_amp=True,
    )

    logger.info(f"訓練完成。")