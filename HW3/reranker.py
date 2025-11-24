# reranker.py
import os
import sys
import json
import random
import sqlite3
import logging
from typing import Dict, List, Set, Tuple

import faiss
import torch
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from sentence_transformers import LoggingHandler
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import SentenceEvaluator

# -----------------------
# Args
# -----------------------
class Args:
    # data paths
    corpus_path = "data/corpus.txt"   # JSONL with {"id": pid, "text": "..."}
    train_path = "data/train.txt"     # JSONL with {"qid": qid, "rewrite": "..."}
    qrels_path = "data/qrels.txt"     # JSON { qid: { pid: label, ... }, ... }

    # sqlite + faiss (for evaluation retrieval)
    index_folder = "vector_database"
    sqlite_file = "passage_store.db"
    faiss_index_file = "passage_index.faiss"

    # model / output
    retriever_model_name_or_path = "output/retriever"
    reranker_model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    output_dir = "./output/reranker"
    max_len = 512

    # training
    seed = 42
    train_epochs = 1
    eval_steps = 1000
    train_batch_size = 8     # CrossEncoder needs small batch sizes on GPU
    num_neg_per_pos = 1      # 每個正樣本抽多少負樣本
    learning_rate = 2e-5
    warmup_steps = 100

    # eval / retrieval
    top_k = 10
    eval_batch_q = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

args = Args()

# -----------------------
# logging
# -----------------------
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

# -----------------------
# helper: load files
# -----------------------
def load_jsonl_data(filepath: str):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_qrels(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Build training examples for cross-encoder
# -----------------------
def build_crossencoder_examples(
    qids: List[str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    pid2text: Dict[str, str],
    all_pids: List[str],
    num_neg_per_pos: int = 1,
    prefix_query: str = "query: ",
    prefix_passage: str = "passage: "
) -> List[InputExample]:
    """
    For each qid in qids:
      - take one positive pid (first non-zero label)
      - sample `num_neg_per_pos` negative pids randomly from all_pids (exclude positives)
      - produce InputExample(texts=[query_pref, passage_pref], label=1/0)
    """
    examples: List[InputExample] = []
    total_passages = len(all_pids)
    if total_passages == 0:
        logger.warning("No passages available for negative sampling.")
        return examples

    for qid in tqdm(qids, desc="build_examples"):
        query_text = queries.get(qid)
        if not query_text:
            continue
        q_pref = prefix_query + query_text

        pid2lab = qrels.get(qid, {})
        gold_pids = [pid for pid, lab in pid2lab.items() if str(lab) != "0"]
        if not gold_pids:
            # no positive for this qid
            continue
        pos_pid = gold_pids[0]
        pos_text = pid2text.get(pos_pid)
        if not pos_text:
            continue
        pos_pref = prefix_passage + pos_text
        examples.append(InputExample(texts=[q_pref, pos_pref], label=1.0))

        # random negatives (avoid pos_pid and other golds)
        neg_needed = min(num_neg_per_pos, total_passages - len(gold_pids))
        negs = set()
        attempts = 0
        max_attempts = neg_needed * 10 + 100
        while len(negs) < neg_needed and attempts < max_attempts:
            cand = random.choice(all_pids)
            if cand == pos_pid or cand in gold_pids or cand in negs:
                attempts += 1
                continue
            negs.add(cand)
            attempts += 1

        for neg_pid in negs:
            neg_text = pid2text.get(neg_pid)
            if not neg_text:
                continue
            examples.append(InputExample(texts=[q_pref, prefix_passage + neg_text], label=0.0))

    logger.info(f"Built {len(examples)} Cross-Encoder training examples (pos+neg).")
    return examples

# -----------------------
# Evaluate reranker: retrieve with retriever+faiss, then rerank with cross-encoder
# compute mrr@{1,3,5,10} and recall@{1,3,5,10}
# -----------------------
def recall_at_k(retrieved_pids: List[str], gold_pids: Set[str], k: int) -> float:
    topk = retrieved_pids[:k]
    return 1.0 if any(pid in gold_pids for pid in topk) else 0.0

def mrr_at_k(reranked_pids: List[str], gold_pids: Set[str], k: int) -> float:
    for rank, pid in enumerate(reranked_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (rank + 1)
    return 0.0

def evaluate_reranker(
    eval_qids: List[str],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    rowid2pidtext: Dict[int, Tuple[str, str]],
    faiss_index,
    retriever: SentenceTransformer,
    reranker: CrossEncoder,
    top_k: int = 10,
    prefix_query: str = "query: ",
    batch_q: int = 16
) -> Dict[str, float]:
    mrr_acc = {k: 0.0 for k in [1,3,5,10]}
    recall_acc = {k: 0.0 for k in [1,3,5,10]}
    n = 0

    #Added tqdm wrapper for a single evaluation progress bar
    for start in tqdm(range(0, len(eval_qids), batch_q), desc="Evaluating Reranker", leave=False):
        batch_qids = eval_qids[start:start+batch_q]
        batch_queries = [prefix_query + queries[qid] for qid in batch_qids]

        # retrieve via retriever + faiss
        q_embs = retriever.encode(
            batch_queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_q,
            show_progress_bar=False  # Disable spammy inner progress bar
        )
        D, I = faiss_index.search(q_embs, top_k)

        # build candidate lists from rowid2pidtext (in-memory)
        batch_cand_ids = []
        batch_cand_texts = []
        for row in I:
            cand_ids = []
            cand_texts = []
            for rid in row.tolist():
                rid = int(rid)
                tup = rowid2pidtext.get(rid)
                if tup is None:
                    continue
                pid, text = tup
                cand_ids.append(pid)
                cand_texts.append(text)
            batch_cand_ids.append(cand_ids)
            batch_cand_texts.append(cand_texts)

        # flatten pairs for reranker scoring
        flat_pairs = []
        idx_slices = []
        cursor = 0
        for qtext, cand_texts in zip(batch_queries, batch_cand_texts):
            n_c = len(cand_texts)
            if n_c == 0:
                idx_slices.append((cursor, cursor))
            else:
                flat_pairs.extend(list(zip([qtext] * n_c, cand_texts)))
                idx_slices.append((cursor, cursor + n_c))
                cursor += n_c

        flat_scores = []
        if flat_pairs:
            flat_scores = reranker.predict(flat_pairs, batch_size=64, show_progress_bar=False)

        # compute metrics per query
        for b_idx, qid in enumerate(batch_qids):
            gold_pids = {pid for pid, lab in qrels.get(qid, {}).items() if str(lab) != "0"}
            low, high = idx_slices[b_idx]
            if low == high:
                # no candidates
                for k in mrr_acc:
                    mrr_acc[k] += 0.0
                    recall_acc[k] += 0.0
                n += 1
                continue
            scores = flat_scores[low:high]
            cand_ids = batch_cand_ids[b_idx]
            reranked = sorted(zip(scores, cand_ids), key=lambda x: x[0], reverse=True)
            reranked_pids = [pid for _, pid in reranked]
            for k in [1,3,5,10]:
                mrr_acc[k] += mrr_at_k(reranked_pids, gold_pids, k)
                recall_acc[k] += recall_at_k(reranked_pids, gold_pids, k)
            n += 1

    if n == 0:
        return {}
    metrics = {}
    for k in [1,3,5,10]:
        metrics[f"mrr@{k}"] = mrr_acc[k] / n
        metrics[f"recall@{k}"] = recall_acc[k] / n
    return metrics


# -----------------------
# Custom Evaluator Class
# -----------------------
class CustomRerankerEvaluator(SentenceEvaluator):
    """
    Wraps the evaluate_reranker function to be used with CrossEncoder.fit
    """
    def __init__(
        self,
        eval_qids: List[str],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        rowid2pidtext: Dict[int, Tuple[str, str]],
        faiss_index,
        retriever: SentenceTransformer,
        top_k: int,
        batch_q: int,
        prefix_query: str = "query: ",
        name: str = "eval"
    ):
        self.eval_qids = eval_qids
        self.queries = queries
        self.qrels = qrels
        self.rowid2pidtext = rowid2pidtext
        self.faiss_index = faiss_index
        self.retriever = retriever
        self.top_k = top_k
        self.batch_q = batch_q
        self.prefix_query = prefix_query
        self.name = name

    def __call__(self, model: CrossEncoder, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training by model.fit()
        """
        if epoch != -1:
            if steps != -1:
                logger.info(f"Evaluating reranker at epoch {epoch}, step {steps}...")
            else:
                logger.info(f"Evaluating reranker at end of epoch {epoch}...")
        else:
            logger.info("Evaluating reranker at start of training...")

        # Run the existing evaluation function
        metrics = evaluate_reranker(
            eval_qids=self.eval_qids,
            queries=self.queries,
            qrels=self.qrels,
            rowid2pidtext=self.rowid2pidtext,
            faiss_index=self.faiss_index,
            retriever=self.retriever,
            reranker=model,  # Pass the model being trained
            top_k=self.top_k,
            prefix_query=self.prefix_query,
            batch_q=self.batch_q
        )

        logger.info(f"Evaluation metrics (epoch {epoch}, step {steps}): {json.dumps(metrics, indent=2)}")

        # The 'fit' function requires a single score to be returned for save_best_model
        # We'll use mrr@10
        return metrics.get(f"mrr@{args.top_k}", 0.0)


# -----------------------
# main
# -----------------------
def main():
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Loading data...")
    corpus_data = load_jsonl_data(args.corpus_path)
    train_data = load_jsonl_data(args.train_path)
    qrels_data = load_qrels(args.qrels_path)

    # corpus: pid -> text
    corpus = {p["id"]: p["text"] for p in corpus_data if "id" in p and "text" in p}

    # queries: qid -> rewrite or query
    queries = {}
    for q in train_data:
        qid = q.get("qid") or q.get("id")
        if not qid: continue
        text = q.get("rewrite") or q.get("query") or q.get("question") or q.get("text")
        if text:
            queries[str(qid)] = text

    # relevant docs: qid -> set(pid)
    relevant_docs = {
        qid: set(pids.keys())
        for qid, pids in qrels_data.items()
        if qid in queries and pids
    }

    logger.info(f"Loaded {len(corpus)} passages, {len(queries)} queries, {len(relevant_docs)} qid->pids mappings.")

    # split 90% train / 10% eval (by qid)
    all_qids = list(relevant_docs.keys())
    random.shuffle(all_qids)
    split_idx = int(len(all_qids) * 0.9)
    train_qids = all_qids[:split_idx]
    eval_qids = all_qids[split_idx:]
    logger.info(f"QIDs total {len(all_qids)} | train {len(train_qids)} | eval {len(eval_qids)}")

    # preload passages from sqlite into memory for fast negative sampling & eval
    sqlite_path = os.path.join(args.index_folder, args.sqlite_file)
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"sqlite file not found: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT rowid, pid, text FROM passages").fetchall()
    rowid2pidtext: Dict[int, Tuple[str, str]] = {}
    pid2text: Dict[str, str] = {}
    all_pids: List[str] = []
    for rid, pid, text in rows:
        rowid2pidtext[int(rid)] = (pid, text)
        pid2text[pid] = text
        all_pids.append(pid)
    logger.info(f"Preloaded {len(all_pids)} passages from sqlite into memory.")

    # build training examples
    logger.info("Building Cross-Encoder training examples (random negatives)...")
    train_examples = build_crossencoder_examples(
        train_qids,
        queries,
        qrels_data,
        pid2text,
        all_pids,
        num_neg_per_pos=args.num_neg_per_pos
    )
    if not train_examples:
        logger.error("No training examples were generated. Exiting.")
        sys.exit(1)

    # create CrossEncoder
    logger.info(f"Loading CrossEncoder: {args.reranker_model_name_or_path}")
    reranker = CrossEncoder(args.reranker_model_name_or_path, device=args.device, max_length=args.max_len)

    # DataLoader (must use reranker.smart_batching_collate)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=reranker.smart_batching_collate
    )

    # load faiss index and retriever for evaluation
    faiss_path = os.path.join(args.index_folder, args.faiss_index_file)
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"faiss index file not found: {faiss_path}")
    faiss_index = faiss.read_index(faiss_path)

    retriever = SentenceTransformer(args.retriever_model_name_or_path, device=args.device)
    retriever.max_seq_length = 512

    # create evaluator
    logger.info("Creating custom evaluator...")
    evaluator = CustomRerankerEvaluator(
        eval_qids=eval_qids,
        queries=queries,
        qrels=qrels_data,
        rowid2pidtext=rowid2pidtext,
        faiss_index=faiss_index,
        retriever=retriever,
        top_k=args.top_k,
        batch_q=args.eval_batch_q,
        prefix_query="query: "
    )

    # train with evaluator
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting training for {args.train_epochs} epochs...")

    reranker.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,                 # <--- Pass the evaluator
        epochs=args.train_epochs,           # <--- Train for all epochs
        evaluation_steps=args.eval_steps,   # <--- Evaluate every N steps
        warmup_steps=args.warmup_steps,
        output_path=args.output_dir,        # <--- Save best model to root output dir
        save_best_model=True,               # <--- Enable saving best model based on evaluator score
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate}
    )

    conn.close()
    logger.info("Done.")

if __name__ == "__main__":
    main()