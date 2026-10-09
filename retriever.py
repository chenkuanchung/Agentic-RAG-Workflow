# retriever.py

import os
import httpx
from typing import Dict, Any, List

from qdrant_client import QdrantClient, models
from langchain_openai import OpenAIEmbeddings
from fastembed import SparseTextEmbedding

# 引入全新的全域狀態結構
from state import AgentState

# ==================================================
# 🌐 初始化檢索模型與資料庫連線
# ==================================================
COLLECTION_NAME = "enterprise_knowledge_base_v2"

# 1. 稠密向量 (Dense) 模型 - Qwen3-Embedding-4B
embeddings_model = OpenAIEmbeddings(
    openai_api_key="EMPTY",
    openai_api_base="https://agentickm.phison.com:8298/v1",
    model="Qwen3-Embedding-4B",
    http_client=httpx.Client(verify=False, timeout=30.0)
)

# 2. 稀疏向量 (Sparse / BM25) 模型
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# 3. Qdrant 連線
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=30.0)

# ==================================================
# ⚙️ 檢索與重排參數設定
# ==================================================
RECALL_POOL_SIZE = 300  # 單一路徑 RRF 召回數量
FINAL_TOP_K = 30        # 全域重排後截斷的最高分 Chunk 數量

RERANKER_API_URL = "https://agentickm.phison.com:8297/v1/rerank"
RERANKER_MODEL = "Qwen3-Reranker-4B"
reranker_client = httpx.Client(verify=False, timeout=45.0)

def rerank_documents(query: str, documents: list[str]) -> list[dict]:
    """呼叫 Cross-Encoder API 對文件進行重新評分排序"""
    if not documents:
        return []
        
    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "documents": documents
    }
    
    try:
        response = reranker_client.post(RERANKER_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if "results" in result:
            return result["results"]
        elif isinstance(result, list): 
            return result
        else:
            print(f"⚠️ [Reranker] 未知的 API 回傳格式: {result.keys()}")
            return []
    except Exception as e:
        print(f"❌ [Reranker API 錯誤]: {e}")
        return []

# ==================================================
# 🚀 核心節點：三合一全域檢索器
# ==================================================
def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    全域超級檢索器：
    1. 多路並發打撈 (Scatter): 遍歷所有 search_queries 發動 Qdrant RRF。
    2. 全域去重 (Deduplicate): 跨路徑彙整，確保送給 Reranker 的 Chunk 不重複。
    3. 全域重排 (Rerank): 用 `original_question` 進行打分，截斷 Top K。
    4. 萃取 DSID: 提取前 K 高分的獨立 DSID 列表供後續全文抓取使用。
    """
    print(f"\n🗄️ [Retriever] 啟動全域混合檢索與上下文重組...")

    expected_gold_dsids = state.get("expected_gold_dsids", [])
    original_question = state.get("original_question", "")
    search_queries = state.get("search_queries", {})
    
    if not search_queries:
        print("   ⚠️ 沒有收到檢索策略，直接退件。")
        return {"status": "failed", "error_history": ["Retriever: search_queries is empty."]}
        
    unique_chunks_map = {}
    
    # =========================================================
    # 🌟 Step 1: 多路打撈與全域去重 (Retrieve & Deduplicate)
    # =========================================================
    for query_id, query_data in search_queries.items():
        hyDE_statement = query_data.get("hypothetical_statement", original_question)
        bm25_keywords = query_data.get("bm25_keywords", [])
        bm25_query_str = " ".join(bm25_keywords)
        
        print(f"   -> 📡 [{query_id}] 執行雙引擎 RRF 檢索...")
        
        try:
            dense_vector = embeddings_model.embed_query(hyDE_statement)
            sparse_result = list(bm25_model.query_embed(bm25_query_str))[0]
            sparse_vector = models.SparseVector(
                indices=[int(idx) for idx in sparse_result.indices],
                values=[float(val) for val in sparse_result.values]
            )

            response = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="", limit=RECALL_POOL_SIZE*3),
                    models.Prefetch(query=sparse_vector, using="bm25-sparse", limit=RECALL_POOL_SIZE*3)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=RECALL_POOL_SIZE, 
                with_payload=True 
            )
            
            duplicate_count = 0
            for hit in response.points:
                payload = hit.payload or {}
                dsid = payload.get("dsid", "Unknown")
                chunk_idx = payload.get("chunk_index", 0)
                
                # 使用 dsid + chunk_index 作為唯一識別碼
                uid = f"{dsid}_{chunk_idx}"
                
                if uid not in unique_chunks_map:
                    unique_chunks_map[uid] = {
                        "uid": uid,
                        "dsid": dsid,
                        "chunk_index": chunk_idx,
                        "content": payload.get("content", ""),
                        "qdrant_score": hit.score, # 保留 Qdrant 分數作為 fallback
                        "file_path": payload.get("file_path", "Unknown")
                    }
                else:
                    duplicate_count += 1
                    
            print(f"      (撈回 {len(response.points)} 筆，過濾掉 {duplicate_count} 筆重複 Chunk)")
            
        except Exception as e:
            print(f"      ⚠️ [{query_id}] 檢索發生錯誤: {e}")

    all_unique_chunks = list(unique_chunks_map.values())
    print(f"   -> 🎯 全域去重完畢！共計蒐集到 {len(all_unique_chunks)} 筆獨立 Chunk。")

    # [追蹤點 1] 檢查 Qdrant 召回池
    if expected_gold_dsids:
        found_in_recall = [dsid for dsid in expected_gold_dsids if any(c["dsid"] == dsid for c in all_unique_chunks)]
        if found_in_recall:
            print(f"      🟢 [漏斗追蹤 1] Qdrant 成功撈回預期黃金文件: {found_in_recall}")
        else:
            print(f"      🔴 [漏斗追蹤 1] 警告！Qdrant 召回池中未包含黃金文件: {expected_gold_dsids}")

    if not all_unique_chunks:
        return {"status": "failed", "error_history": ["Retriever: 所有檢索路徑均無回傳資料。"]}

    # =========================================================
    # 🌟 Step 2: Global Cross-Encoder 重排 (Reranking)
    # =========================================================
    print(f"   -> 🧠 將 {len(all_unique_chunks)} 筆 Chunk 送交 Cross-Encoder 進行全域重排...")
    
    docs_for_reranker = [c["content"] for c in all_unique_chunks]
    # ⚠️ 極度重要：這裡強制使用 original_question 進行打分，確保不偏離主軸
    rerank_results = rerank_documents(original_question, docs_for_reranker)
    
    if rerank_results:
        try:
            score_key = "relevance_score" if "relevance_score" in rerank_results[0] else "score"
            for res in rerank_results:
                idx = res.get("index")
                if idx is not None and idx < len(all_unique_chunks):
                    all_unique_chunks[idx]["rerank_score"] = res.get(score_key, 0)
        except Exception as e:
            print(f"      ⚠️ Reranker 解析失敗 ({e})，使用 Qdrant RRF 分數保底。")
            for c in all_unique_chunks:
                c["rerank_score"] = c["qdrant_score"]
    else:
        print("      ⚠️ Reranker 未回傳結果，使用 Qdrant RRF 分數保底。")
        for c in all_unique_chunks:
            c["rerank_score"] = c["qdrant_score"]

    # 排序並截斷 Top K
    all_unique_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    top_k_chunks = all_unique_chunks[:FINAL_TOP_K]
    print(f"   -> ✅ 重排完成！已截斷保留前 {len(top_k_chunks)} 筆最高分 Chunk。")

    # [追蹤點 2] 檢查 Reranker 截斷池
    if expected_gold_dsids:
        found_in_top_k = [dsid for dsid in expected_gold_dsids if any(c["dsid"] == dsid for c in top_k_chunks)]
        if found_in_top_k:
            print(f"      🟢 [漏斗追蹤 2] Reranker 成功將黃金文件保留在 Top {FINAL_TOP_K}: {found_in_top_k}")
        else:
            print(f"      🔴 [漏斗追蹤 2] 警告！Reranker 截斷後，黃金文件被擠出 Top {FINAL_TOP_K}: {expected_gold_dsids}")

    # =========================================================
    # 🌟 Step 3: 提取 DSID 供下游迴圈使用 (舊的重組邏輯已刪除)
    # =========================================================
    unique_dsids = list(dict.fromkeys([c["dsid"] for c in top_k_chunks]))

    return {
        "dsid_queue": unique_dsids,        # 供後續迴圈使用
        "status": "reading"
    }