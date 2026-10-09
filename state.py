# state.py

import operator
from typing import TypedDict, Annotated, List, Dict, Any

class AgentState(TypedDict):
    """
    集中式 Enterprise RAG 狀態機
    從「多任務平行」改為「多路檢索、集中審判」的流水線架構。
    """
    # ================= 1. 核心輸入 =================
    original_question: str
    atomic_questions: List[str]  # Analyzer 動態產出的原子問題 (數量不限)
    
    # ================= 2. 檢索策略與資料 =================
    # 將原始問題與所有原子問題的 HyDE/BM25 策略集中存放
    search_queries: Dict[str, Any] 
    
    # ================= 3. 審判與打撈 =================
    dsid_queue: List[str]            # 從 Retriever 拿到的 Top 20 獨立 DSID 列表
    global_expanded_docs: Dict[str, str]
    
    # ================= 4. 最終產出 =================
    evidence_reports: List[Dict[str, Any]]
    final_answer: str
    
    # ================= 5. 系統控制 =================
    # 狀態流轉: pending -> searching -> reading -> needs_expansion -> verifying -> completed/failed
    status: str
    global_step_count: Annotated[int, operator.add]
    gold_dsids: List[str]                # 真正被大斥候採用的黃金文件
    error_history: Annotated[List[str], operator.add]
    expected_gold_dsids: List[str]