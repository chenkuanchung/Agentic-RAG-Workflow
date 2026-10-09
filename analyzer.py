# analyzer.py

import json
import re
import time
import httpx
from typing import Dict, Any

from state import AgentState

http_client = httpx.Client(verify=False, timeout=999.0)
VLLM_API_URL = "https://agentickm.phison.com:8299/v1/responses" 
MODEL_NAME = "GLM-4.7-Flash" 

def invoke_llm_json(prompt: str, system_prompt: str, temperature: float = 0.0, max_retries: int = 1) -> str:
    """輕量級的 LLM 呼叫函式，專門要求回傳 JSON 格式"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    
    for attempt in range(max_retries):
        try:
            response = http_client.post(VLLM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if "output" in result and isinstance(result["output"], list):
                final_text = ""
                for item in result["output"]:
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        for content_item in item.get("content", []):
                            if content_item.get("type") == "output_text":
                                final_text += content_item.get("text", "")
                if final_text.strip():
                    return final_text
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            raise ValueError("API 回傳成功，但找不到有效的文字內容")
        except Exception as e:
            print(f"🔴 [LLM API 錯誤] (第 {attempt+1}/{max_retries} 次): {e}")
            time.sleep(2)
    return None

def analyzer_node(state: AgentState) -> Dict[str, Any]:
    """
    分析師節點：將原始問題拆解為多個原子問題 (不再使用 sub_tasks)
    """
    original_question = state["original_question"]
    print(f"🧠 [Analyzer] 收到原始問題:\n   {original_question}")

    system_prompt = """You are an expert Question Analyzer for an advanced enterprise RAG system.
Your ONLY task is to decompose a complex user query into a list of independent, self-contained "atomic questions".

CRITICAL RULES:
1. DECOMPOSITION: If the user asks for a comparison (e.g., A vs B), you MUST split it into at least two questions: "What is A?" and "What is B?".
2. CONTEXT INHERITANCE: Every atomic question MUST retain ALL environmental constraints and context from the original query.
3. PREVENT OVER-FRAGMENTATION: Do not break a single consecutive sequence of events or a specific bug description into general, disconnected questions.
4. SELF-CONTAINED: Replace pronouns (it, this, they) with the exact entity names.
5. If the original question is already focused on a specific issue, return it as a single-item list.

OUTPUT FORMAT:
You MUST return ONLY a valid JSON array of strings. Do not include markdown code blocks.
"""

    response_text = invoke_llm_json(
        prompt=f"Original Question: {original_question}", 
        system_prompt=system_prompt
    )

    atomic_questions = []
    if response_text is None:
        print(f"⚠️ [Analyzer 警告] API 失敗，啟動保底機制。")
        atomic_questions = [original_question]
    else:
        cleaned_text = re.sub(r"```json|```", "", response_text).strip()
        try:
            atomic_questions = json.loads(cleaned_text)
            if not isinstance(atomic_questions, list) or len(atomic_questions) == 0:
                raise ValueError("LLM 沒有回傳有效的 JSON 陣列。")
        except Exception as e:
            print(f"⚠️ [Analyzer 警告] JSON 解析失敗 ({e})，退回不拆解模式。")
            atomic_questions = [original_question]

    print(f"\n🎯 [Analyzer] 成功產出 {len(atomic_questions)} 個原子任務:")
    for i, aq in enumerate(atomic_questions):
        print(f"   [{i+1}] {aq}")

    # 直接回傳 atomic_questions 陣列，並將狀態推進到 formulating
    return {
        "atomic_questions": atomic_questions,
        "status": "formulating"
    }

# ==================================================
# 🧪 測試腳本區塊 (併發壓力測試)
# ==================================================
if __name__ == "__main__":
    import concurrent.futures

    test_questions = [
        #   "What was the high-percentile latency concern reported after a smoke test of about 50 concurrent streaming chat sessions for Satellite Grove?",
        #   "What is the company policy for how long contractor access should last by default before it expires, according to the access and permissions playbook?",
        #   "On managed iPhones on corporate Wi-Fi, why do long-lived server push text streams sometimes reconnect after the app is backgrounded and then end with a few seconds of silence and a cut-off final JSON fragment instead of a clean end marker?",
        #   "After our single sign on provider URL changed, access approvals look successful in the internal approval UI but engineers still get a forbidden error when trying to open privileged operational tools; what caused the approval to not actually translate into permissions?",
        #   "For the Q2 milestone scope they locked in the chat, which items were pushed to v1 versus kept in the MVP?",
        #   "What is the rollback plan for adaptive batching and INT8 models?"
        #   "What is the recommended step-by-step rollback and traffic split plan to mitigate correctness issues and tail latency spikes after a customer enabled adaptive batching together with an INT8 quantized model on their dedicated production route?",
          "In the March 2026 fix for Hosted API issues where SSE streams were truncated/partial behind corporate proxies (e.g., NGINX/Envoy idle timeouts), what resume identifier format did we choose for Last-Event-ID-style reconnects, and per our 2026 partial-result checkpointing/resume standard what are the default checkpoint emission cadence and checkpoint TTL?"
    ]

    def run_analyzer_test(question):
        """包裝單一測試任務以供執行緒池呼叫"""
        mock_state = {
            "original_question": question,
            "atomic_questions": [],
            "search_queries": {},
            "global_expanded_docs": {},
            "evidence_reports": [],
            "final_answer": "",
            "status": "pending",
            "global_step_count": 0,
            "gold_dsids": [],
            "error_history": []
        }
        
        result = analyzer_node(mock_state)
        return result

    print(f"🚀 啟動併發測試：準備同時發送 {len(test_questions)} 個問題給 LLM API...")
    start_time = time.time()

    # 建立擁有 5 個 worker 的執行緒池，同時發起請求
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # map 函式會把 test_questions 派發給 5 個 worker 併發執行
        results = list(executor.map(run_analyzer_test, test_questions))

    end_time = time.time()
    
    print(f"✅ 併發測試執行完畢！總耗時: {end_time - start_time:.2f} 秒")
    
    # 稍微印出第一筆結果的狀態，確認新架構回傳無誤
    if results:
        print("\n🔍 測試結果抽查 (第一題的最終回傳狀態):")
        print(f"狀態 (status): {results[0].get('status')}")
        print(f"產出的原子問題數量: {len(results[0].get('atomic_questions', []))}")
