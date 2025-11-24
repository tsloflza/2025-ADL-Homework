# rl.py
import numpy as np
import json, faiss, torch
import gymnasium as gym
from gymnasium import spaces
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
import os
import sqlite3
import gc
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import get_inference_system_prompt, get_inference_user_prompt, parse_generated_answer

# -----------------------
# Load HF Token
# -----------------------
load_dotenv()
hf_token = os.getenv("hf_token")
login(token=hf_token)

# -----------------------
# Args (Hardcoded)
# -----------------------
class Args:
    # --- Data Paths ---
    data_folder = "./data"
    passage_file = "corpus.txt"
    index_folder = "./vector_database"
    index_file = "passage_index.faiss"
    sqlite_file = "passage_store.db"
    test_data_path = "./data/test_open.txt"
    qrels_path = "./data/qrels.txt"

    # --- Model Paths ---
    retriever_model_path = "output/retriever"
    reranker_model_path = "output/reranker"
    generator_model = "Qwen/Qwen3-1.7B"
    scorer_model = "sentence-transformers/all-MiniLM-L6-v2"

    output_dir = "./output/rl"

    # --- RAG Parameters ---
    TOP_K = 10         # Retriever 檢索 K 篇
    GEN_MAXLEN = 512   # LLM 生成最大長度
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- RL Parameters ---
    # Top_M 的最大值 (Action 空間會是 1 到 TOP_K)
    MAX_TOP_M = TOP_K  
    
    # 由於 LLM 很慢，我們只用少量數據
    N_TEST_CASES_FOR_RL = 100  # 只取前 n 筆 test data 來訓練/測試 RL
    TOTAL_TIMESTEPS = 100     # RL 總共只訓練 n 步 (即跑 n 次 RAG 流程)


args = Args()

# -----------------------
# RAG RL Environment
# -----------------------
class RAGEnv(gym.Env):
    """
    一個自定義的 Gym 環境，用於 RAG 流程中的 Top_M 選擇。

    - Observation: Reranker 排序後的 Top_K 個分數 (shape: [K,])
    - Action: 選擇 Top_M (Discrete(K)，動作 0 對應 M=1, 動作 K-1 對應 M=K)
    - Reward: 生成答案與黃金答案的 cosine similarity
    """
    metadata = {'render_modes': []}

    def __init__(self, args: Args):
        super(RAGEnv, self).__init__()
        
        self.args = args
        self.top_k = args.TOP_K
        
        # --- 1. Define Action and Observation Spaces ---
        
        # Action: 選擇 M=1 到 M=TOP_K。
        # 動作 0 -> M=1, 動作 1 -> M=2, ..., 動作 9 -> M=10
        self.action_space = spaces.Discrete(self.top_k) 
        
        # Observation: Top_K 個 reranker 分數
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.top_k,), dtype=np.float32
        )
        
        print("Initializing RAGEnv... Loading all models.")
        
        # --- 2. Load all models and data ---
        self.retriever = SentenceTransformer(args.retriever_model_path, device=args.DEVICE)
        self.reranker = CrossEncoder(args.reranker_model_path, device=args.DEVICE)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            args.generator_model, 
            torch_dtype="auto", 
            device_map="auto"
        )
        self.scorer = SentenceTransformer(args.scorer_model, device=args.DEVICE)

        # --- 3. Load DB connection ---
        sqlite_path = os.path.join(args.index_folder, args.sqlite_file)
        self.conn = sqlite3.connect(sqlite_path)
        self.cur = self.conn.cursor()
        
        self.index = faiss.read_index(os.path.join(args.index_folder, args.index_file))

        # --- 4. Load Test Data ---
        self.tests = self._load_test_data(args.test_data_path, args.qrels_path)
        # 只使用一小部分數據
        if args.N_TEST_CASES_FOR_RL > 0:
            self.tests = self.tests[:args.N_TEST_CASES_FOR_RL]
        print(f"Loaded {len(self.tests)} test cases for RL.")
        self.current_test_index = 0
        
        # 儲存當前 episode 的數據
        self.current_reranked_data = []
        self.current_query = ""
        self.current_gold_answer = ""
        self.current_qid = ""

    def _load_test_data(self, test_path, qrels_path):
        # (Helper) Load qrels
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels = json.load(f)
        qid2gold = {}
        for qid, pid2lab in qrels.items():
            gold = {pid for pid, lab in pid2lab.items() if str(lab) != "0"}
            qid2gold[qid] = gold

        # (Helper) Load test queries
        tests = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                qid = obj.get("qid")
                query = obj.get("rewrite")
                gold_answer = (obj.get("answer")).get("text", "")
                gold_pids = qid2gold.get(qid, set())
                # 必須要有黃金答案才能計算 reward
                if query and gold_answer:
                    tests.append({
                        "qid": qid, 
                        "query": query, 
                        "gold_answer": gold_answer, 
                        "gold_pids": gold_pids
                    })
        return tests

    def _get_obs_and_reranked_data(self):
        """
        執行 Retrieve 和 Rerank 步驟，以獲取狀態 (Observation)。
        """
        # 1. 獲取當前測試案例
        test_case = self.tests[self.current_test_index]
        self.current_test_index = (self.current_test_index + 1) % len(self.tests) # 循環
        
        self.current_query = test_case["query"]
        self.current_gold_answer = test_case["gold_answer"]
        self.current_qid = test_case["qid"]
        
        # 2. Retrieve (Top_K)
        prefix_query = "query: " + self.current_query
        q_emb = self.retriever.encode(
            [prefix_query], convert_to_numpy=True, normalize_embeddings=True
        )
        D, I = self.index.search(q_emb, self.top_k)
        
        rowids = I[0].tolist()
        need_rowids = set(int(rid) for rid in rowids)
        
        # 3. Get Passages from DB
        rowid2pt = {}
        if need_rowids:
            placeholders = ",".join(["?"] * len(need_rowids))
            sql = f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})"
            rows = self.cur.execute(sql, tuple(need_rowids)).fetchall()
            rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

        cand_ids, cand_texts = [], []
        for rid in rowids:
            tup = rowid2pt.get(int(rid))
            if tup is None: continue
            cand_ids.append(tup[0])
            cand_texts.append(tup[1])
            
        # 4. Rerank
        reranked_data = []
        scores_list = []
        
        if cand_texts:
            pairs = list(zip([self.current_query] * len(cand_texts), cand_texts))
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            
            reranked_data = sorted(
                zip(scores, cand_ids, cand_texts), 
                key=lambda x: x[0], 
                reverse=True
            )
            scores_list = [float(s) for s, _, _ in reranked_data]

        # 儲存 rerank 結果以供 step() 使用
        self.current_reranked_data = reranked_data
        
        # 5. Build Observation
        # 用 0.0 填充不足 TOP_K 的部分
        scores_padded = scores_list + [-100.0] * (self.top_k - len(scores_list))
        obs = np.array(scores_padded[:self.top_k], dtype=np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 獲取下一個 query 的 rerank 分數作為初始狀態
        observation = self._get_obs_and_reranked_data()
        info = {}
        
        return observation, info

    def step(self, action):
        # 1. Map Action to TOP_M
        # action 0 -> M=1, ..., action 9 -> M=10
        top_m = int(action) + 1 
        
        # 2. Select (Top_M)
        context_list = [text for _, _, text in self.current_reranked_data[:top_m]]
        
        # 3. Generate (LLM)
        messages = [
            {"role": "system", "content": get_inference_system_prompt()},
            {"role": "user", "content": get_inference_user_prompt(self.current_query, context_list)}
        ]
        
        self.llm_tokenizer.padding_side = "left"
        rendered_prompt = self.llm_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        inputs = self.llm_tokenizer(
            [rendered_prompt], padding=True, return_tensors="pt"
        ).to(self.llm_model.device)
        
        with torch.no_grad():
            outs = self.llm_model.generate(**inputs, max_new_tokens=args.GEN_MAXLEN)
        
        decoded = self.llm_tokenizer.batch_decode(outs, skip_special_tokens=True)
        pred_ans = parse_generated_answer(decoded[0].strip())

        # 4. Calculate Reward
        try:
            emb_res = self.scorer.encode(
                [pred_ans], convert_to_tensor=True, normalize_embeddings=True
            )
            emb_gold = self.scorer.encode(
                [self.current_gold_answer], convert_to_tensor=True, normalize_embeddings=True
            )
            
            score = util.cos_sim(emb_res, emb_gold)[0][0]
            reward = float(score)

        except Exception as e:
            print(f"Error calculating reward: {e}")
            print(f"Pred Ans: {pred_ans}")
            print(f"Gold Ans: {self.current_gold_answer}")
            reward = 0.0

        # 5. Return values
        # 一個 episode = 一個 query。所以 step 完就 terminated。
        terminated = True
        truncated = False 
        info = {
            "qid": self.current_qid,
            "top_m_selected": top_m,
            "reward": reward,
            "pred_ans": pred_ans,
            "gold_answer": self.current_gold_answer
        }
        
        # 下一個 state 在 reset() 時獲取，這裡返回一個 dummy state
        dummy_obs = np.zeros(self.top_k, dtype=np.float32)
        
        return dummy_obs, reward, terminated, truncated, info

    def close(self):
        print("Closing RAGEnv...")
        self.conn.close()
        del self.retriever, self.reranker, self.llm_model, self.llm_tokenizer, self.scorer
        gc.collect()
        torch.cuda.empty_cache()

# -----------------------
# Main Training & Eval Loop
# -----------------------
def main():
    print("--- Starting RL for RAG Top-M Tuning ---")
    
    # 1. Initialize Environment
    # SB3 需要一個 VecEnv (即使只有一個環境)
    print("Creating environment...")
    try:
        env = DummyVecEnv([lambda: RAGEnv(args)])
    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("Please check all model paths and data paths in the Args class.")
        return

    # 2. Initialize Agent (A2C is simple and fast)
    # "MlpPolicy" 適用於我們向量化的 observation (分數列表)
    print("Initializing A2C agent...")
    model = A2C(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=args.output_dir,
        gamma=0.9, # 在 episodic 任務中 gamma 不太重要，但還是設一下
        n_steps=5  # 每 5 步更新一次
    )

    # 3. Train the Agent
    print(f"Starting training for {args.TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=args.TOTAL_TIMESTEPS, log_interval=1)
        save_path = os.path.join(args.output_dir, "rag_a2c_top_m_agent")
        model.save(save_path)
        print("Training finished and model saved.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        env.close()
        return

    # 4. Evaluate the Trained Agent
    print("\n--- Evaluating trained agent ---")
    obs = env.reset()
    
    total_rewards = []
    actions_taken = []
    
    # 評估 N_TEST_CASES_FOR_RL 步
    for i in tqdm(range(len(env.envs[0].tests)), desc="Evaluating"):
        # deterministic=True: 選擇模型認為最好的動作，而不是採樣
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, info = env.step(action)
        
        total_rewards.append(reward[0])
        actions_taken.append(info[0]["top_m_selected"])
        
        if (i+1) % 10 == 0:
             print(f"Eval step {i+1}: QID {info[0]['qid']}")
             print(f"  Action (Top_M): {info[0]['top_m_selected']}, Reward: {reward[0]:.4f}")

    # 5. Show Results
    print("\n--- Evaluation Results ---")
    print(f"Average Reward (Similarity): {np.mean(total_rewards):.4f}")

    print(f"\nTop_M Sequence (n={len(actions_taken)}):")
    print(actions_taken)
    
    # 統計 Agent 選擇的 Top_M 分布
    action_counts = np.bincount(actions_taken)
    print("\nAction (Top_M) Distribution:")
    for m_value, count in enumerate(action_counts):
        if count > 0:
            print(f"  Top_M = {m_value}: {count} times")

    env.close()
    print("Done.")

if __name__ == "__main__":
    main()