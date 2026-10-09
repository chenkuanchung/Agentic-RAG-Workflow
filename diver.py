# diver.py

import os
from typing import Dict, Any
import threading
_index_lock = threading.Lock()
from state import AgentState

# ==================================================
# 🗄️ 實體檔案設定與全文獲取工具 (O(1) 極速版)
# ==================================================
RAW_DATA_DIR = os.path.abspath(os.getenv("RAW_DATA_DIR", "../all_documents"))

# 建立全域 DSID 索引表 (In-Memory Index)
DSID_TO_PATH_CACHE = {}
_is_index_built = False

def build_file_index():
    """
    在程式首次啟動時，遍歷一次文件目錄，建立 { dsid: 完整絕對路徑 } 的 HashMap。
    """
    global _is_index_built
    if _is_index_built: return
    with _index_lock:
        if _is_index_built: return  # 雙重檢查
        print("⚙️ [系統初始化] 正在建立實體檔案路徑索引...")
    count = 0
    
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.startswith("dsid_"):
                # 利用切片 file[:37] 抓出 DSID
                dsid_key = file[:37] 
                DSID_TO_PATH_CACHE[dsid_key] = os.path.join(root, file)
                count += 1
                
    print(f"✅ [系統初始化] 索引建立完成！共記錄了 {count} 份文件的實體路徑。")
    _is_index_built = True

def fetch_full_document_by_dsid(dsid: str) -> str:
    """透過 O(1) 的記憶體索引瞬間取得檔案路徑並讀取全文"""
    build_file_index() 
    
    target_file = DSID_TO_PATH_CACHE.get(dsid)
    
    if not target_file:
        print(f"      ❌ 索引表中找不到 DSID 為 {dsid} 的檔案！")
        return ""
        
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"      ❌ 讀取文件 {target_file} 時發生錯誤: {e}")
        return ""

# ==================================================
# 🤿 潛水員節點核心邏輯
# ==================================================
def diver_node(state: AgentState) -> Dict[str, Any]:
    """
    純 I/O 節點 (批次打撈版)：
    一次性將 dsid_queue 裡的所有文件實體全文撈出，放上全局黑板。
    """
    dsid_queue = state.get("dsid_queue", [])
    
    print(f"\n🤿 [Diver] 潛水員啟動 (準備批次打撈 {len(dsid_queue)} 篇實體文件)...")
    
    if not dsid_queue:
        print("   ⚠️ [警告] 空的 dsid_queue，無任務可做。")
        return {
            "status": "failed", 
            "error_history": ["Diver: dsid_queue is empty."]  # 修正這裡：拿掉 error_history +
        }
        
    expanded_docs = {}
    new_errors = []
    
    for dsid in dsid_queue:
        full_text = fetch_full_document_by_dsid(dsid)
        if full_text.strip():
            expanded_docs[dsid] = full_text
        else:
            # 【關鍵改良】軟性防呆：印出警告並記錄，但不中斷整個迴圈
            print(f"      ⚠️ [警告] 實體檔案 {dsid} 遺失或無法讀取，已略過。")
            new_errors.append(f"Diver: 實體檔案 {dsid} 遺失或無法讀取。")
            
    print(f"   ✅ 打撈完成！成功取得 {len(expanded_docs)} 篇全文，準備交給大斥候進行平行審查。")
    
    return {
        "global_expanded_docs": expanded_docs,
        "error_history": new_errors, 
        "status": "verifying"
    }

# ==================================================
# 🧪 測試腳本區塊
# ==================================================
if __name__ == "__main__":
    import time
    # 若要在本機測試時強制指定路徑，可以在執行前設定環境變數
    # os.environ["RAW_DATA_DIR"] = r"C:\Users\USER\Desktop\AgenticKM-EnterpriseRAG\all_documents"
    
    # 模擬 Retriever 傳來的新版狀態
    mock_state = {
        "original_question": "Test single-document iterative retrieval?",
        "atomic_questions": [],
        "search_queries": {},
        "rebuilt_context": [],
        "dsid_queue": [
            "dsid_a592f24d2f3b4c21ba507b20c832d069", 
            "dsid_5e87f57f0e424a3a9e79a3032d63eab8"
        ],
        "current_doc_index": 0, # 測試第一圈
        "is_sufficient": False,
        "global_expanded_docs": {}, 
        "evidence_reports": [],
        "final_answer": "",
        "status": "reading",
        "global_step_count": 0,
        "gold_dsids": [],
        "error_history": []
    }
    
    start_time = time.time()
    result_state = diver_node(mock_state)
    end_time = time.time()
    
    print("\n" + "="*50)
    print(f"✅ 測試完成！總耗時: {end_time - start_time:.4f} 秒")
    print(f"最終狀態 (status): {result_state.get('status')}")
        
    print("\n📝 檢查全局黑板 (global_expanded_docs) 上的內容:")
    for dsid, text in result_state.get("global_expanded_docs", {}).items():
        print(f"- {dsid}: 成功加載 {len(text)} 字元 | 預覽: {text[:100].replace(chr(10), ' ')}...")