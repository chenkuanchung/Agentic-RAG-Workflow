# graph.py

import time
from langgraph.graph import StateGraph, START, END

# 引入全域狀態結構
from state import AgentState

# 引入我們精心打造的護法節點
from analyzer import analyzer_node
from formulator import formulator_node
from retriever import retriever_node
from diver import diver_node
from deep_verifier import deep_verifier_node
from chief_judge import chief_judge_node

def check_retriever_result(state: AgentState):
    """判斷檢索器是否有撈回資料"""
    if not state.get("dsid_queue"):
        return "chief_judge" # 沒東西直接去結案道歉
    return "diver"

def build_graph():
    print("🧠 [System] 正在編譯 LangGraph 工作流 (平行展開版)...")
    
    workflow = StateGraph(AgentState)

    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("formulator", formulator_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("diver", diver_node)
    workflow.add_node("deep_verifier", deep_verifier_node)
    workflow.add_node("chief_judge", chief_judge_node)

    # 建立直通車邊界 (不再有迴圈)
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "formulator")
    workflow.add_edge("formulator", "retriever")
    
    workflow.add_conditional_edges(
        "retriever",
        check_retriever_result,
        {
            "diver": "diver",
            "chief_judge": "chief_judge"
        }
    )
    
    # 潛水員撈完全部文件 -> 送交高階幕僚平行處理
    workflow.add_edge("diver", "deep_verifier")
    
    # 高階幕僚平行處理完 -> 直接送交大審判長
    workflow.add_edge("deep_verifier", "chief_judge")
    
    workflow.add_edge("chief_judge", END)

    app = workflow.compile()
    print("✅ [System] LangGraph 平行工作流編譯完成！\n")
    return app

# ==================================================
# 🧪 端到端 (End-to-End) 系統整合測試 (支援批次測試)
# ==================================================
if __name__ == "__main__":
    import time
    
    app = build_graph()
    
    # 支援多題測試，將前五題解除註解
    test_questions = [
        "What are the default size limits for file uploads and total request size for the new multipart upload support on the OpenAI-compatible API endpoints?",
        "What is the name of the new metric added so SRE can track when server-side streaming sessions get finalized due to hitting the time limit?",
        "What are the acceptance criteria for the project introducing an algorithm to generate interactive UI color states and a Kappa-style elevation scale for dense table and grid components?",
        "In the meeting about onboarding a SaaS product to Google Cloud Marketplace, what did the GCP team recommend for handling delays where a new subscription entitlement is not immediately available during the customer onboarding flow?",
        "What failover sequence and recovery targets did MedThink specify for handling an EU region outage, including any limits on how long traffic can shift to the US?",
    ]

    expected_gold_dsids_list = [
        ["dsid_ae068ee4aa9640159427cd941bef0238"],
        ["dsid_9550250a59e74f1bbd5612480b2e7100"],
        ["dsid_3fd6af404fae48e6b8ea5a57875ef78f"],
        ["dsid_6c4c1c875e704f09b4d791d64d7bc7e5"],
        ["dsid_8e838ab6a98f4cbcb672d41f210ff89c"]
    ]
    
    print(f"\n🚀 [系統啟動] 準備進行批次測試，共計 {len(test_questions)} 題。\n")
    
    for i, question in enumerate(test_questions, 1):

        # 安全的取值方式，防止 IndexError
        if (i - 1) < len(expected_gold_dsids_list):
            current_expected_dsids = expected_gold_dsids_list[i-1]
        else:
            current_expected_dsids = []
        
        print("=" * 70)
        print(f"▶️ [測試題 {i}/{len(test_questions)}] 正在處理：")
        print(f"{question}")
        print("=" * 70)
        
        initial_state = {
            "original_question": question,
            "atomic_questions": [],
            "search_queries": {},
            "dsid_queue": [],
            "global_expanded_docs": {},
            "evidence_reports": [],
            "final_answer": "",
            "status": "pending",
            "global_step_count": 0,
            "gold_dsids": [],
            "error_history": [],
            "expected_gold_dsids": current_expected_dsids
        }
        
        start_time = time.time()
        
        try:
            # 啟動 LangGraph
            final_state = app.invoke(initial_state)
            end_time = time.time()
            
            print("\n")
            print(f"🏁 [第 {i} 題 執行完畢] 總耗時: {end_time - start_time:.2f} 秒")
            print(f"🧩 產出的原子問題數: {len(final_state.get('atomic_questions', []))}")
            print(f"🧩 產出的檢索策略數: {len(final_state.get('search_queries', {}))}")
            print(f"🏆 命中黃金文件: {final_state.get('gold_dsids', [])}")
            
            # 【修正點】: 改用 global_expanded_docs 字典的長度來計算實際讀取的文章數
            print(f"🔄 總計閱讀全文數量: {len(final_state.get('global_expanded_docs', {}))}")
            
            # 如果有退件或警告紀錄，印出來方便除錯
            error_history = final_state.get('error_history', [])
            if error_history:
                print(f"⚠️ 異常/退件紀錄: {error_history}")
                
            print(f"\n📝 最終答案:\n{final_state.get('final_answer', '')}\n")
            
        except Exception as e:
            end_time = time.time()
            print(f"\n❌ [第 {i} 題 執行失敗] 發生未預期錯誤 (耗時: {end_time - start_time:.2f} 秒): {e}\n")