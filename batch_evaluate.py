# batch_evaluate.py

import json
import argparse
import concurrent.futures
import time
import os
from typing import Dict, Any
from tqdm import tqdm

# 匯入我們打磨好的 Agentic RAG 狀態機
from graph import build_graph
# 建立並編譯 Graph 實體
app = build_graph()

def process_single_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    處理單一問題的獨立任務函數
    """
    # 支援不同資料集的常見欄位命名
    q_id = item.get("question_id", item.get("id", "unknown_id"))
    question_text = item.get("question", item.get("query", ""))
    expected_doc_ids = item.get("expected_doc_ids", [])

    # 建立乾淨的初始狀態
    initial_state = {
        "question_id": q_id,
        "original_question": question_text,
        "semantic_queries": [],
        "terminology_queries": [],
        "evidence_queries": [],
        "target_source_types": [],
        "current_retrieved_docs": [],
        "useful_docs": [],
        "verified_facts": "",
        "is_sufficient": False,
        "missing_information": "",
        "search_strategy": "none",
        "needs_context_expansion": False,
        "target_expand_dsid": "",
        "target_expand_chunk_idx": 0,
        "model_reasoning": "",
        "final_answer": "",
        "final_document_ids": [],
        "retry_count": 0,
        "is_unanswerable": False,
        "expansion_cursors": {},
        "expected_gold_dsids": expected_doc_ids
    }

    try:
        # 啟動 Agent 推理
        final_state = app.invoke(initial_state)
        
        retrieved_docids = final_state.get("gold_dsids", [])
        
        # [可選] 計算命中率 (Hit Rate)：檢查預期的黃金文件是否被成功撈出
        hit_count = sum(1 for expected_id in expected_doc_ids if expected_id in retrieved_docids)
        is_hit = hit_count > 0 and len(expected_doc_ids) > 0

        return {
            "question_id": q_id,
            "answer": final_state.get("final_answer", "System error: No answer generated."),
            "document_ids": retrieved_docids,
            "metadata": { # 可選，加入一些幫助 debug 的中繼資訊
                "expected_doc_ids": expected_doc_ids,
                "is_hit": is_hit,
                "hit_count": hit_count
            }
        }
    except Exception as e:
        # 確保就算單題崩潰，也不會搞垮整個批次任務
        return {
            "question_id": q_id,
            "answer": f"Agent Pipeline Crashed: {str(e)}",
            "document_ids": [],
            "metadata": {"error": str(e)}
        }

def main():
    # 1. 設定命令列參數 (Argparse)
    parser = argparse.ArgumentParser(description="Agentic RAG 批量非同步測試腳本 (智慧覆蓋版)")
    parser.add_argument("--input", type=str, default="../questions.jsonl", help="輸入的問題檔 (JSONL 格式)")
    parser.add_argument("--output", type=str, default="answers.jsonl", help="輸出的解答檔 (JSONL 格式)")
    # 注意：題號從 1 開始
    parser.add_argument("--start", type=int, default=1, help="從第幾題開始測 (題號從 1 開始)")
    parser.add_argument("--end", type=int, default=None, help="測到第幾題為止 (包含此題)")
    parser.add_argument("--workers", type=int, default=1, help="多管齊下的併發數量 (Threads)")
    args = parser.parse_args()

    # 2. 讀取題目
    print(f"📂 正在讀取題目檔: {args.input}")
    questions = []
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ 找不到輸入檔案: {args.input}")
        return

    # 3. 陣列切片 (控制測試範圍，並處理從 1 開始的邏輯)
    total_questions = len(questions)
    
    # 將人類的 1-based 轉為 Python 的 0-based index
    start_idx = max(0, args.start - 1)
    
    if args.end is not None:
        end_idx = min(total_questions, args.end)
    else:
        end_idx = total_questions
        
    # 防止範圍不合理
    if start_idx >= end_idx:
        print("⚠️ 測試範圍錯誤，請確認 --start 和 --end 參數。")
        return

    target_questions = questions[start_idx:end_idx]
    target_count = len(target_questions)
    
    print(f"🎯 總題數: {total_questions} | 本次將測試題號 {start_idx + 1} 至 {end_idx} (共 {target_count} 題)")
    print(f"🚀 啟動多執行緒併發測試 (Workers: {args.workers})...")
    print("-" * 50)

    # 4. 讀取既有的解答紀錄 (用於「已存在覆蓋，不存在追加」邏輯)
    existing_answers = {}
    if os.path.exists(args.output):
        print(f"🔍 發現既有解答檔 {args.output}，載入歷史紀錄中...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "question_id" in record:
                            existing_answers[record["question_id"]] = record
                    except json.JSONDecodeError:
                        pass # 忽略壞掉的行

    start_time = time.time()
    total_hits = 0

    # 5. 啟動 ThreadPoolExecutor 與進度條
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 將任務提交給執行緒池
        future_to_item = {executor.submit(process_single_question, item): item for item in target_questions}
        
        # 使用 tqdm 顯示進度條
        with tqdm(total=target_count, desc="驗證進度", unit="題") as pbar:
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                q_id = item.get("question_id", "unknown")
                try:
                    result = future.result()
                    
                    # 統計 Hit Rate
                    if result.get("metadata", {}).get("is_hit", False):
                        total_hits += 1
                        
                except Exception as exc:
                    print(f"\n⚠️ 題目 {q_id} 發生未預期的系統級崩潰: {exc}")
                    result = {
                        "question_id": q_id, 
                        "answer": f"Critical Thread Error: {exc}", 
                        "document_ids": []
                    }
                
                # 做好一題就即時更新、排序並存檔 (進度防丟失機制)
                existing_answers[result["question_id"]] = result
                sorted_answers = sorted(existing_answers.values(), key=lambda x: x.get("question_id", ""))
                
                # 每次有一題完成，就立刻覆寫檔案，確保中斷也不會遺失進度
                with open(args.output, "w", encoding="utf-8") as out_f:
                    for record in sorted_answers:
                        # 只提取評測系統需要的三個欄位
                        clean_record = {
                            "question_id": record.get("question_id"),
                            "answer": record.get("answer"),
                            "document_ids": record.get("document_ids", [])
                        }
                        out_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                
                # 更新進度條
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print("-" * 50)
    
    # 6. 迴圈結束，只需印出最終結果
    sorted_answers = sorted(existing_answers.values(), key=lambda x: x.get("question_id", ""))
    print(f"🎉 測試完成！總紀錄數: {len(sorted_answers)}")
    print(f"📂 結果已安全儲存且排序完畢至 {args.output}")
    if target_count > 0:
        print(f"🎯 本次執行命中率 (Hit Rate): {total_hits}/{target_count} ({total_hits/target_count*100:.1f}%)")
        print(f"⏱️ 總耗時: {elapsed_time:.2f} 秒 (平均單題: {elapsed_time/target_count:.2f} 秒)")

if __name__ == "__main__":
    main()