# Agentic-RAG-Workflow ⚖️

> 一個基於 LangGraph 構建的企業級 Agentic RAG 工作流，具備平行文件審查、動態提早退場機制以及嚴格的防幻覺審判架構。

## 🚀 專案概述 (Overview)

在正式環境中打造可靠的 Agentic RAG 系統，絕不只是單純把 Prompt 串接起來而已。它需要處理不穩定的 API、克服 LLM 上下文限制，並從根本上防止模型產生幻覺。

**Agentic-RAG-Workflow** 正是為了解決這些企業級痛點而生。本系統摒棄了完全自主但難以預測的 Agent 框架，轉而採用基於 LangGraph 的嚴格 **「大審判長 / 高階幕僚 (Chief Judge / Deep Verifier)」** 雙層架構。我們犧牲了非結構化的 LLM 漫遊自主性，換取了企業系統最需要的：高度確定性、低延遲與絕對的事實準確度。

## ✨ 核心亮點 (Key Features)

- **多階段審查架構 (Multi-Stage Verification Architecture):**
  - 🧙‍♂️ **高階幕僚 (Deep Verifiers):** 以併發方式平行審閱檢索到的文件，嚴格檢驗該文件是否能為使用者的問題提供明確、完整的解答。
  - 👨‍⚖️ **大審判長 (Chief Judge):** 綜合萃取出的事實、解決衝突資訊，並撰寫最終的詳盡解答。內建嚴格的「逃生出口 (Escape Hatch)」機制，在資料不足時能安全宣告「查無資訊」，徹底杜絕 LLM 腦補幻覺。
- **動態提早退場與併發控制 (Dynamic Early Exit & Concurrency Control):** 一旦收集到足夠的 `TRUE` 證據報告，系統即刻中斷剩餘的驗證流程。此機制能有效防止 LLM API Timeout 並大幅降低伺服器算力負載。
- **雙引擎混合檢索 (Hybrid Search):** 採用 **Qdrant** 結合稠密向量 (Qwen3 Dense) 與稀疏向量 (BM25 Sparse)，並透過 RRF (Reciprocal Rank Fusion) 演算法實現最佳化的文件召回率。
- **高容錯 JSON 解析 (Bulletproof JSON Parsing):** 系統底層整合 `json-repair`，能優雅處理並自動修復 LLM 產出的未跳脫雙引號 (Unescaped Quotes) 或格式毀損的 JSON 回應，達成極致的穩定性。

## 🛠️ 技術棧 (Tech Stack)

- **核心框架:** Python, [LangGraph](https://python.langchain.com/docs/langgraph)
- **向量資料庫:** [Qdrant](https://qdrant.tech/) (Docker 部署)
- **嵌入模型 (Embeddings):** FastEmbed (BM25) + OpenAI 相容 API 介面 (Qwen3)
- **關鍵套件:** `json-repair`, `httpx`, `tqdm`

## 📊 資料集與評測基準 (Dataset & Evaluation Benchmark)

本專案針對 **EnterpriseRAG-Bench** 進行建置與測試。若要執行評測並重現本系統：

1. **取得資料集**: 
   請前往官方 Benchmark 儲存庫：[onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)。
   下載來源文件後，請將其放置於 `../all_documents/` 目錄下（或在執行 `data_loader.py` 時透過 `--docs_dir` 參數指定自訂路徑）。

2. **準備評測題庫**:
   從該 Benchmark 儲存庫取得測試資料集（例如 `test_set.jsonl`）。

## ⚙️ 快速開始 (Quick Start)

### 1. 安裝相依套件
```bash
pip install -r requirements.txt
```

### 2. 啟動向量資料庫
請確認已安裝 Docker，接著啟動 Qdrant 服務：
```bash
docker-compose up -d
```

### 3. 匯入文件與建表
執行 Data Loader 進行文件切片 (Chunking)、生成雙向量，並於背景建立 Qdrant 檢索索引：
```bash
python data_loader.py --docs_dir ../all_documents --workers 16
```

### 4. 執行管線與批次評測
執行批次評測腳本，讓測試題庫通過完整的 Agentic Workflow。系統將自動處理文件檢索、平行審查與最終答案生成。
```bash
python batch_evaluate.py
```
最終生成的答案與系統圈選出的「黃金文件 (Gold Documents)」將自動儲存於 `answers.jsonl` 中。

## 📂 專案結構 (Project Structure)

- `graph.py` / `state.py`: LangGraph 狀態機定義與工作流路由控制。
- `data_loader.py`: 負責文件切片與 Qdrant 雙向量寫入（含背景非同步建表優化）。
- `formulator.py`: 查詢擴展 (Query Expansion) 與全域檢索策略生成。
- `retriever.py` / `diver.py`: 執行 RRF 混合檢索與實體檔案打撈。
- `deep_verifier.py`: 執行平行文件審查與動態提早退場機制。
- `chief_judge.py`: 執行最終決策、資訊衝突解決與完美解答撰寫。
- `analyzer.py`: LLM API 封裝介面與 JSON 容錯修復工具。

---
*Built for robustness, concurrency, and factual integrity.*
