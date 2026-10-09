# formulator.py

import json
import re
from typing import Dict, Any

from state import AgentState
from analyzer import invoke_llm_json

def formulator_node(state: AgentState) -> Dict[str, Any]:
    """
    全域查詢生成器節點：
    將「原始問題」以及 Analyzer 拆出來的「原子問題」，統一轉換為 HyDE 虛擬聲明與 BM25 關鍵字。
    """
    print(f"🔍 [Formulator] 啟動全域檢索策略生成...")

    original_question = state.get("original_question", "")
    atomic_questions = state.get("atomic_questions", [])
    
    search_queries = {}

    system_prompt = """You are an expert Search Query Formulator for an enterprise RAG system.
Your task is to convert a question into two highly optimized search components for a Hybrid Search engine.

CRITICAL RULES:
1. HYPOTHETICAL STATEMENT: Transform the question into a declarative sentence that looks exactly like a snippet from an official technical document. Leave it open-ended.
2. BM25 KEYWORDS: Extract ONLY the most critical entities, specific project codes, parameter names, and technical nouns. Return them as a JSON array of strings.
3. ENTITY ANCHORING: Never drop, translate, or modify specific names or technical parameters.

OUTPUT FORMAT:
Return ONLY a valid JSON object containing exactly two keys: "hypothetical_statement" and "bm25_keywords". No markdown.
"""

    def generate_strategy(query_id: str, question_text: str):
        print(f"   -> 正在處理 [{query_id}]: {question_text}")
        response_text = invoke_llm_json(
            prompt=f"Question: {question_text}",
            system_prompt=system_prompt
        )
        
        try:
            cleaned_text = re.sub(r"```json|```", "", response_text).strip()
            search_components = json.loads(cleaned_text)
            
            if "hypothetical_statement" not in search_components or "bm25_keywords" not in search_components:
                raise KeyError("Missing required keys.")
        except Exception as e:
            print(f"      ⚠️ [警告] JSON 解析失敗，啟動 Fallback。")
            search_components = {
                "hypothetical_statement": question_text,
                "bm25_keywords": [w for w in question_text.replace("?", "").split() if len(w) > 3]
            }
            
        search_queries[query_id] = search_components
        print(f"      -> 🧠 HyDE: {search_components['hypothetical_statement']}")
        print(f"      -> 🔑 BM25: {search_components['bm25_keywords']}")

    # 1. 處理原始問題 (Macro Query / 保底檢索)
    if original_question:
        generate_strategy("original_macro_query", original_question)

    # 2. 處理所有原子問題 (Micro Queries / 擴大召回)
    for idx, atomic_q in enumerate(atomic_questions):
        generate_strategy(f"atomic_query_{idx+1}", atomic_q)

    print(f"\n✅ [Formulator] 策略生成完畢！共產出 {len(search_queries)} 組檢索條件，準備交給超級檢索器。")

    # 回傳 search_queries 字典，並將流程推進至 Retriever
    return {
        "search_queries": search_queries,
        "status": "searching"
    }

# ==================================================
# 🧪 測試腳本區塊 (單獨測試 Formulator 邏輯)
# ==================================================
if __name__ == "__main__":
    import time
    
    # 模擬從 Analyzer 傳來的全新狀態帳本
    mock_state = {
        "original_question": "After our single sign on provider URL changed, access approvals look successful in the internal approval UI but engineers still get a forbidden error when trying to open privileged operational tools; what caused the approval to not actually translate into permissions?",
        "atomic_questions": [
            "Why did the access approval fail to translate into permissions after the SSO provider URL changed, resulting in the internal approval UI showing success while engineers received a forbidden error when trying to open privileged operational tools?"
        ],
        "search_queries": {},
        "global_expanded_docs": {},
        "evidence_reports": [],
        "final_answer": "",
        "status": "formulating",
        "global_step_count": 0,
        "gold_dsids": [],
        "error_history": []
    }
    
    print("🚀 啟動 Formulator 測試...")
    start_time = time.time()
    
    # 執行 Formulator
    result_state = formulator_node(mock_state)
    
    end_time = time.time()
    
    print(f"\n✅ 測試執行完畢！總耗時: {end_time - start_time:.2f} 秒")
    print(f"最終狀態: {result_state.get('status')}")
    
    print("\n🔍 檢查產出的檢索策略：")
    for query_id, query_data in result_state.get("search_queries", {}).items():
        print(f"\n[{query_id}]")
        print(f"   HyDE: {query_data.get('hypothetical_statement')}")
        print(f"   BM25: {query_data.get('bm25_keywords')}")