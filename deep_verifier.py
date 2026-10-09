# deep_verifier.py

import json
import re
import concurrent.futures
from typing import Dict, Any

from state import AgentState
from analyzer import invoke_llm_json

system_prompt = """You are a STRICT Document Verifier. Your task is to compare the "Original Question" against the "provided Document". 
You must find ANY differences. Documents that look similar but have different details MUST be rejected.
If the provided Document fails the comparison, you must set `match_status` to "false". 
If the provided Document perfectly matches, you must extract the exact details into `semi_final_answer`.

You MUST write down your step-by-step thinking in the `reasoning` field before making a decision. Follow these exact steps:

--- WORKFLOW ---

STEP 1: QUESTION TARGETS (List ALL you can get)
Find the exact requirements in the "Original Question":
- What EXACT feature, algorithm, or project name is requested?
- What EXACT team or person is requested?
- What EXACT context, event, or format is requested?
- What EXACT constraints, environments, or specific conditions are mentioned?

STEP 2: DOCUMENT FACTS (List ALL you can get)
Find the exact facts written in the "provided Document":
- What EXACT feature, algorithm, or project name does the Document talk about?
- What EXACT team or person does the Document talk about?
- What is the EXACT context of the Document? [EXAMPLE: Is it a meeting note, a PR draft, or a final specification?]
- What EXACT constraints, environments, or specific conditions does the Document mention?

STEP 3: STRICT COMPARISON (Find the Differences/Discrepancies)
Compare Step 1 against Step 2 item by item. You must look for any mismatch in logical entities, exact naming, scope, and explicit constraints.
- Rule 1: ENTITY & GRANULARITY MATCH. The concepts must be technically equivalent in scope. [EXAMPLE: A "compatibility layer" (architecture) is NOT the same as an "API endpoint" (network location). If the granularity differs, it is a FATAL MISMATCH.]
- Rule 2: EXACT NAMING & NO SYNONYMS. Do not guess or treat similar-looking names as equal. If a name, version, or detail is slightly different, it fails immediately. [EXAMPLE: "Alpha-style" vs "Alpha-lite" is a FATAL MISMATCH.]
- Rule 3: EXPLICIT EVIDENCE REQUIRED. If the Original Question contains a specific requirement, constraint, or environment, the Document MUST EXPLICITLY state it. If the Document is silent on a constraint, or you have to "assume" it applies, it is a FATAL MISMATCH.
- Rule 4: NO CONTEXT MISMATCH. [EXAMPLE: If Step 1 asks about a "meeting", but Step 2 is a "specification", this is a FATAL MISMATCH].

STEP 4: ANSWER CHECK
ONLY do this step if Step 3 has ZERO mismatches. Check if the provided Document has the answer:
- Does the Document contain ALL the facts needed to answer the Original Question?
- Or does the Document only contain a small piece of the answer?

--- DECISION MATRIX (match_status) ---
- "false": If Step 3 finds ANY mismatch, OR if the Document cannot answer the Original Question at all.
- "true": If Step 3 is a PERFECT match, AND Step 4 shows the Document FULLY answers the Original Question.
- "could_be_useful": If Step 3 is a PERFECT match, BUT Step 4 shows the Document only PARTIALLY answers the Original Question.

--- EXTRACTION RULE (semi_final_answer) ---
If `match_status` is "true" or "could_be_useful", extract the exact answer:
1. Write the exact numbers, limits, criteria, or decisions found in the Document.
2. Write the exact conditions. [EXAMPLE: If the Document says "for hosted only", you MUST write "for hosted only" in your answer].
If `match_status` is "false", leave `semi_final_answer` empty ("").

OUTPUT FORMAT:
Return ONLY a valid JSON object. Do not output anything outside of the JSON object:
{
  "reasoning": "<Step 1: Question Targets> ... <Step 2: Document Facts> ... <Step 3: Strict Comparison> ... <Step 4: Answer Check> ... <Final Conclusion> ...",
  "match_status": "true | false | could_be_useful",
  "semi_final_answer": "<Extracted comprehensive details from the DOCUMENT, OR leave an empty string If `match_status` is "false">"
}
"""

def evaluate_single_doc(dsid: str, text: str, original_question: str) -> Dict[str, Any]:
    """封裝單一文件的審閱邏輯，供執行緒池調用"""
    prompt = (
        f"Original Question: {original_question}\n"
        f"\n[DOCUMENT TO REVIEW: {dsid}]\n{text}"
    )
    
    response_text = invoke_llm_json(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0
    )
    
    try:
        if not response_text:
            raise ValueError("Empty response")
            
        cleaned_text = re.sub(r"```json|```", "", response_text).strip()
        judgment = json.loads(cleaned_text)
        
        match_status = judgment.get("match_status", "false").lower()
        
        # 只有當狀態不是 false，才需要把資料傳回去
        if match_status in ["true", "could_be_useful"]:
            return {
                "dsid": dsid,
                "match_status": match_status,
                "reasoning": judgment.get("reasoning", ""),
                "semi_final_answer": judgment.get("semi_final_answer", "")
            }
        return None
    except Exception as e:
        print(f"      ⚠️ [{dsid}] 判決書解析失敗跳過 ({e})")
        return None

def deep_verifier_node(state: AgentState) -> Dict[str, Any]:
    original_question = state.get("original_question", "")
    global_expanded_docs = state.get("global_expanded_docs", {})
    evidence_reports = state.get("evidence_reports", []) # 保留先前的報告(若有)
    
    doc_count = len(global_expanded_docs)
    print(f"\n🧙‍♂️ [Deep Verifier] 高階幕僚登入！啟動 {doc_count} 影分身平行審閱...")

    if doc_count == 0:
        return {"status": "chief_judge"}

    # =========================================================
    # ⚡ 核心併發區塊 (平行發送 API 請求 + Early Exit 機制)
    # =========================================================
    true_reports = []
    useful_reports = []
    MAX_REPORTS_TO_JUDGE = 5  # 定義我們需要的完美報告數量

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 1. 提交所有任務
        future_to_dsid = {
            executor.submit(evaluate_single_doc, dsid, text, original_question): dsid 
            for dsid, text in global_expanded_docs.items()
        }
        
        # 2. 收集完成的結果
        for future in concurrent.futures.as_completed(future_to_dsid):
            result = future.result()
            if result:
                match_status = result['match_status']
                if match_status == "true":
                    true_reports.append(result)
                else:
                    useful_reports.append(result)
                
                print(f"   ✅ [命中目標] 發現 {match_status.upper()} 報告: {result['dsid']}")
                print(f"      📝 [幕僚推論]: {result['reasoning']}")
                preview_answer = result['semi_final_answer'].replace('\n', ' ')
                print(f"      💡 [申論預覽]: {preview_answer}...\n")

                # 動態提早退場機制：
                # 條件 1：(True + Useful) 總數湊滿指定數量 (預設 5 份)
                # 條件 2：這幾份裡面，至少要有 1 份是 True 報告 (保底解答品質)
                if (len(true_reports) + len(useful_reports)) >= MAX_REPORTS_TO_JUDGE and len(true_reports) >= 1:
                    print(f"   🛑 [動態退場] 已收集 {len(true_reports)} 份 TRUE 與 {len(useful_reports)} 份 USEFUL 報告，滿足底線要求，終止審閱！")
                    # 取消還在執行緒池排隊的剩餘任務 (需 Python 3.9+)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # 將先前的報告與這次收集到的報告合併
    all_reports = evidence_reports + true_reports + useful_reports
    
    # 🛡️ 重新嚴格排序：確保所有的 'true' 永遠排在 'could_be_useful' 前面
    all_reports.sort(key=lambda x: 0 if x["match_status"] == "true" else 1)
    
    # 截斷保留 Top 5 (去重複功能可視需求加入，目前先照舊)
    sorted_and_limited_reports = all_reports[:MAX_REPORTS_TO_JUDGE]
    
    print(f"   🎯 平行審閱完畢！經提早退場與截斷後，送交 {len(sorted_and_limited_reports)} 份情報給審判長。")

    return {
        "evidence_reports": sorted_and_limited_reports,
        "status": "chief_judge" 
    }

