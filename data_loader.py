# data_loader.py

import os
import re
import uuid
import time
import argparse
import threading
from pathlib import Path
from dotenv import load_dotenv
import httpx
from tqdm import tqdm
import warnings
import concurrent.futures

# 💡 解決 Windows [WinError 1314]：強制關閉 HuggingFace 的符號連結機制
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_CACHE_SYMLINKS"] = "1"

# Qdrant 相關套件
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from fastembed import SparseTextEmbedding

# 關閉 SSL 警告
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

load_dotenv()

# ==========================================
# ⚙️ 核心設定與全域開關
# ==========================================
INJECT_METADATA = True
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "enterprise_knowledge_base_v2" # 預設使用 V2 混合檢索庫
SPARSE_VECTOR_NAME = "bm25-sparse"
VECTOR_SIZE = 2560  

# 🚨 定義全域中斷訊號 (Fail-Fast Switch)
abort_event = threading.Event()

# 全域共用 QdrantClient (預設 Timeout 設為 120 秒)
global_qdrant_client = QdrantClient(url=QDRANT_URL, timeout=120.0)

# 初始化 Dense Embedding 模型
embeddings_model = OpenAIEmbeddings(
    openai_api_key="EMPTY",
    openai_api_base="https://agentickm.phison.com:8298/v1",
    model="Qwen3-Embedding-4B",
    http_client=httpx.Client(verify=False, timeout=120.0)
)

# 初始化 Sparse Embedding 模型 (本地 CPU 運算)
print("📦 正在載入 FastEmbed BM25 模型...")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def init_qdrant_collection():
    """初始化 Qdrant Collection，包含雙向量 Schema 與 Payload 索引 (含背景建表)"""
    if global_qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"ℹ️ Collection '{COLLECTION_NAME}' 已存在，將繼續追加/覆寫資料。")
    else:
        print(f"⚠️ 正在建立全新的混合檢索 Collection '{COLLECTION_NAME}'...")
        global_qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        print(f"✅ 成功創建 Collection (支援 Dense + Sparse 雙架構)")
        
        print(f"⚡ 正在為 Metadata 建立 Payload 索引...")
        
        # 1. 建立 source_type 索引 (同步)
        global_qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_type",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        # 2. 建立 dsid 索引 (非同步背景處理，避免海量資料 Timeout)
        global_qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="dsid",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=False  
        )
        
        # 3. 建立 chunk_index 索引 (非同步背景處理)
        global_qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="chunk_index",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=False  
        )
        print(f"✅ 索引建立完成！(dsid 與 chunk_index 任務已派發至背景處理)")

def process_batch(batch_texts, batch_payloads):
    """【工人任務】處理批次寫入，具備斷點續傳與防護機制"""
    if abort_event.is_set() or not batch_texts:
        return

    # 斷點續傳核心：預先計算 UUID
    batch_ids = []
    for payload in batch_payloads:
        unique_string = f"{payload['dsid']}_{payload['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
        batch_ids.append(point_id)

    # 批次查詢是否已存在
    try:
        existing_points = global_qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch_ids,
            with_payload=False,  
            with_vectors=False   
        )
        existing_ids = {point.id for point in existing_points}
    except Exception as e:
        print(f"\n⚠️ 查詢已存在 UUID 時發生錯誤，將預設重新覆寫: {e}")
        existing_ids = set()

    new_texts = []
    new_payloads = []
    new_ids = []
    
    for text, payload, point_id in zip(batch_texts, batch_payloads, batch_ids):
        if point_id not in existing_ids:
            new_texts.append(text)
            new_payloads.append(payload)
            new_ids.append(point_id)
            
    if not new_texts:
        return

    # 呼叫 API 並寫入 Qdrant
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 1. 遠端計算 Dense Vectors
            dense_embeddings = embeddings_model.embed_documents(new_texts)
            
            # 2. 本地計算 Sparse Vectors
            sparse_embeddings = list(bm25_model.embed(new_texts))
            
            points = []
            for payload, dense_vec, sparse_vec, point_id in zip(new_payloads, dense_embeddings, sparse_embeddings, new_ids):
                # 強制轉換 numpy 型別以防 JSON 序列化失敗
                sparse_indices = [int(idx) for idx in sparse_vec.indices]
                sparse_values = [float(val) for val in sparse_vec.values]
                
                points.append(
                    PointStruct(
                        id=point_id,
                        payload=payload,
                        vector={
                            "": dense_vec, # 預設的未命名 Dense 向量
                            SPARSE_VECTOR_NAME: models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        }
                    )
                )
            
            global_qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            break 
            
        except Exception as e:
            if abort_event.is_set():
                break
                
            if attempt < max_retries - 1:
                print(f"\n⚠️ 批次寫入失敗，正在進行第 {attempt + 2} 次重試... (錯誤: {e})")
                time.sleep(2)  
            else:
                print(f"\n❌ 批次向量化或寫入 Qdrant 失敗，已達最大重試次數: {e}")
                abort_event.set()  
                raise e

def process_and_ingest_documents(target_dir: Path, base_dir: Path, max_workers: int, limit: int = None):
    """主執行緒：負責讀檔、切塊與任務分發"""
    init_qdrant_collection()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )

    all_txt_files = list(target_dir.rglob("*.txt"))
    if limit:
        all_txt_files = all_txt_files[:limit]
    
    if not all_txt_files:
        print(f"⚠️ 在 {target_dir} 中沒有找到任何 .txt 檔案。")
        return

    print(f"🔍 準備處理 [{target_dir.name}] 目錄下的 {len(all_txt_files)} 份文件...")
    print(f"🚀 啟動多執行緒引擎 (工人數: {max_workers})")

    global_batch_texts = []
    global_batch_payloads = []
    
    GLOBAL_MAX_SIZE = 100 
    MAX_QUEUE_SIZE = max_workers * 2
    active_futures = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in tqdm(all_txt_files, desc=f"📦 {target_dir.name} 進度"):
            if abort_event.is_set():
                print("\n🚨 接收到嚴重錯誤廣播，主執行緒立即中斷檔案讀取！")
                break

            rel_path_obj = file_path.relative_to(base_dir)
            source_type = rel_path_obj.parts[0] if len(rel_path_obj.parts) >= 1 else "unknown"
            relative_path = rel_path_obj.as_posix()
            
            filename = file_path.name
            dsid_match = re.match(r"^(dsid_[a-f0-9]+)", filename)
            if not dsid_match: 
                continue
            dsid = dsid_match.group(1)
            doc_title = filename.replace(f"{dsid}__", "").replace(".txt", "").replace("-", " ")

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read().replace('\x00', '')
            except Exception:
                continue

            chunks = text_splitter.split_text(raw_text)
            total_chunks = len(chunks)
            
            for idx, chunk_text in enumerate(chunks):
                if INJECT_METADATA:
                    final_content = (
                        f"[Title: {doc_title}]\n"
                        f"\n"
                        f"[File Path: {relative_path}]\n"
                        f"---\n"
                        f"{chunk_text}"
                    )
                else:
                    final_content = chunk_text
                
                payload = {
                    "dsid": dsid,
                    "doc_title": doc_title,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "source_type": source_type,
                    "file_path": relative_path,
                    "content": final_content
                }
                
                global_batch_texts.append(final_content)
                global_batch_payloads.append(payload)
                
                if len(global_batch_texts) >= GLOBAL_MAX_SIZE:
                    future = executor.submit(
                        process_batch, 
                        global_batch_texts.copy(), 
                        global_batch_payloads.copy()
                    )
                    active_futures.add(future)
                    
                    global_batch_texts.clear()
                    global_batch_payloads.clear()
                    
                    try:
                        if len(active_futures) >= MAX_QUEUE_SIZE:
                            done, active_futures = concurrent.futures.wait(
                                active_futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in done:
                                f.result()  
                    except Exception:
                        abort_event.set()
                        break

        if not abort_event.is_set() and global_batch_texts:
            future = executor.submit(
                process_batch, 
                global_batch_texts.copy(), 
                global_batch_payloads.copy()
            )
            active_futures.add(future)
            
        if not abort_event.is_set():
            print("\n⏳ 檔案讀取完畢！等待所有 API 請求與 Qdrant 寫入完成...")
        
        try:
            for future in concurrent.futures.as_completed(active_futures):
                future.result()
        except Exception:
            abort_event.set()

    if abort_event.is_set():
        print(f"\n💥 [{target_dir.name}] 匯入任務已強制中止！請檢查網路或調降 workers 數量。")
    else:
        print(f"\n✅ [{target_dir.name}] 目錄處理完畢，已全部匯入 Qdrant！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="../all_documents", help="官方文件的根目錄")
    parser.add_argument("--target_folder", type=str, default=None, help="指定要處理的特定子資料夾名稱 (例如: fireflies)")
    parser.add_argument("--workers", type=int, default=16, help="同時發送 API 請求的工作執行緒數量")
    parser.add_argument("--limit", type=int, default=None, help="測試用的檔案數量限制")
    args = parser.parse_args()

    base_dir_path = Path(args.docs_dir)
    
    if args.target_folder:
        target_dir_path = base_dir_path / args.target_folder
        if not target_dir_path.exists() or not target_dir_path.is_dir():
            print(f"❌ 找不到目錄: {target_dir_path}")
            exit(1)
    else:
        target_dir_path = base_dir_path
        print("⚠️ 未指定 --target_folder，將會掃描根目錄下的所有檔案。")

    process_and_ingest_documents(target_dir_path, base_dir_path, args.workers, args.limit)