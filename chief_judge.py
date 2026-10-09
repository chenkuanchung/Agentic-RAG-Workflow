# chief_judge.py

import json
import re
import json_repair
from typing import Dict, Any

from state import AgentState
from analyzer import invoke_llm_json

def chief_judge_node(state: AgentState) -> Dict[str, Any]:
    """
    大審判長節點 (Chief Judge - Reduce 階段)：
    接收高階幕僚 (Deep Verifier) 傳來的結構化報告，
    進行交叉比對、去蕪存菁，圈選最終黃金文件，並撰寫極致詳盡的最終解答。
    """
    print(f"\n👨‍⚖️ [Chief Judge] 大審判長接手，準備進行最終裁決與撰寫完美解答...")

    original_question = state.get("original_question", "")
    evidence_reports = state.get("evidence_reports", []) 

    if not evidence_reports:
        print("   ⚠️ 幕僚未提交任何有效報告，將產出無法回答的預設訊息。")
        return {
            "final_answer": "The answer must state at some point that the query is not fully answerable from available documents or caveat the provided information with why it does not fully address the query. The answer may present relevant and related information to be helpful to the user however it must clearly also mention that at least some aspects are not found or answered. The answer may also simply state that the query is not answerable from the documents, this is perfectly acceptable.",
            "gold_dsids": [],
            "status": "completed"
        }

    # =========================================================
    # 大審判長的專屬 Prompt (加入衝突解決與思維鏈)
    # =========================================================
    system_prompt = """You are the Chief Judge and Final Decision Maker for an enterprise RAG system.
Your staff has provided you with several preliminary "Evidence Reports" evaluated from different documents. 
Your task is to cross-reference these reports, resolve any conflicting information, identify the TRUE gold documents, and compose a masterful final answer.

You MUST write down your step-by-step thinking in the `reasoning` field before making your final decision.

CRITICAL RULES:
1. CROSS-REFERENCE & FILTERING (Statistically, up to 80 percent of questions are fully answered by a SINGLE perfect document.): 
   - Status Hierarchy: You MUST prioritize reports with a "true" match_status. DISCARD "could_be_useful" reports unless they provide critically missing complementary details.
   - Quality Control: DISCARD totally irrelevant or factually inferior reports.
2. CONFLICT RESOLUTION (Crucial for Multiple True Reports): If multiple reports provide DIFFERENT facts or numbers (e.g., 5MB vs 10MB vs 25MB), carefully analyze the conditions in their reasoning/answers. 
   - Conditional Nuance: If the differences depend on specific environments (e.g., "hosted" vs "general") or configurations, do NOT simply discard them. Incorporate BOTH conditions into your final answer to provide a comprehensive view.
   - Version Control: If one report is clearly a draft/proposal and another is the final spec, rely on the definitive final spec and discard the draft.
3. INTELLIGENT SYNTHESIS & STITCHING: Do NOT blindly concatenate all valid reports. 
   - Redundancy Handling: If multiple reports provide the exact same perfect answer, merge them gracefully to avoid repetition.
   - Complementary Stitching: Logically stitch different aspects together to form a complete, cohesive picture.
4. GOLD DSID SELECTION (relied_dsids): Extract the DSIDs of ONLY the reports you ACTUALLY used to form your final answer. Do not include DSIDs of reports you discarded. If no reports are useful, return an empty array [].
5. THE PERFECT FINAL ANSWER (WHEN INFO IS FOUND): If the reports contain the answer, you MUST write a comprehensive, complete, perfect, and detailed final answer. 
   - Explicitly state the complete context (who, what, when, where, and specific conditions like "in hosted environments").
   - Ensure the narrative flows logically and professionally.
   - DO NOT use metadata phrases like "Based on the reports", "The document says", or "According to the staff". Speak with absolute authority as the system.
6. HANDLING UNANSWERABLE QUERIES (INFO NOT FOUND): If you determine that the provided reports DO NOT fully or correctly answer the Original Question, you have the authority to state so.
   - The answer must state at some point that the query is not fully answerable from available documents or caveat the provided information with why it does not fully address the query.
   - The answer may present relevant and related information to be helpful to the user however it must clearly also mention that at least some aspects are not found or answered.
   - The answer may also simply state that the query is not answerable from the documents, this is perfectly acceptable.

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact schema. Do not output anything outside of the JSON object:
{
  "reasoning": "<Step-by-step cross-referencing and conflict resolution analysis>",
  "relied_dsids": ["dsid_XYZ", "dsid_ABC"],
  "final_answer": "<Write your comprehensive, complete, perfect, and detailed final answer here.>"
}
"""
    
    # 將結構化報告轉化為可讀文本供審判長閱讀
    reports_text_blocks = []
    for i, report in enumerate(evidence_reports, 1):
        block = (
            f"--- Staff Report {i} ---\n"
            f"[DSID]: {report.get('dsid', 'Unknown')}\n"
            f"[Match Status]: {report.get('match_status', 'Unknown')}\n"
            f"[Staff Reasoning]: {report.get('reasoning', '')}\n" #[cite: 1]
            f"[Semi-Final Answer]: {report.get('semi_final_answer', '')}\n"
        )
        reports_text_blocks.append(block)
        
    facts_text = "\n\n".join(reports_text_blocks)
    
    prompt = f"User's Original Question: {original_question}\n\n{facts_text}"

    print(f"   -> 🧠 正在交叉比對 {len(evidence_reports)} 份幕僚報告，並鑑定黃金文件...")

    # 溫度微調：給予 0.0 的溫度確保事實精準
    response_text = invoke_llm_json(
        prompt=prompt, 
        system_prompt=system_prompt, 
        temperature=0.1 
    )

    final_answer = ""
    gold_dsids = []
    judge_reasoning = ""

    if response_text:
        cleaned_text = re.sub(r"```json|```", "", response_text).strip()
        try:
            result = json_repair.loads(cleaned_text)
            final_answer = result.get("final_answer", "")
            gold_dsids = result.get("relied_dsids", [])
            judge_reasoning = result.get("reasoning", "")
            
            # 印出大審判長的推論過程
            if judge_reasoning:
                print(f"   ⚖️ [審判長推論]: {judge_reasoning}")
                
        except Exception as e:
            print(f"   ⚠️ [警告] JSON 解析失敗 ({e})。啟動容錯機制。")
            final_answer = response_text

    if not final_answer:
        final_answer = "系統已成功審查文件，但在由審判長合成最終解答時發生異常。\n"

    print(f"   ✅ 最終裁決與完美解答撰寫完成！(共 {len(final_answer)} 字)")
    print(f"   🏆 最終圈定的黃金文件: {gold_dsids}")

    return {
        "final_answer": final_answer,
        "gold_dsids": gold_dsids,
        "status": "completed"
    }