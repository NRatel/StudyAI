# 高级 RAG 技术

> **前置知识**：[Embedding 与向量检索](2_Embedding与向量检索.md)（已理解 Embedding 模型、向量数据库、Chunking 策略）
>
> **本节目标**：掌握 Naive RAG 的典型失败模式及其解决方案——Query 改写/HyDE、Reranker 重排、混合检索、Agentic RAG，以及如何用 RAGAS 等框架系统性评估 RAG 质量。

---

## 1. 直觉与概述

### 1.1 Naive RAG 的瓶颈在哪里？

回顾 Naive RAG 的流程：用户问题 → Embedding → 向量检索 → Top-K 拼接 → LLM 生成。

这个简单流程在以下场景中会失败：

| 失败场景 | 示例 | 根因 |
|---------|------|------|
| **查询模糊** | "那个东西怎么用？" | 查询向量语义不明确 |
| **词汇鸿沟** | 查询用"LLM"，文档写"大语言模型" | Embedding 未充分对齐同义表达 |
| **精确匹配需求** | "error code E4523" | 向量检索擅长语义，不擅长精确匹配 |
| **多跳推理** | "A 公司 CEO 毕业于哪所大学？" | 需要先找 CEO 是谁，再查其教育背景 |
| **噪声干扰** | Top-K 中混入无关文档 | 向量相似 ≠ 真正相关 |

### 1.2 Advanced RAG 优化全景

```
            预检索优化                    检索优化                  后检索优化
         ┌──────────┐              ┌──────────────┐          ┌───────────────┐
         │ Query 改写│              │ 混合检索      │          │ Reranker 重排  │
         │ HyDE     │     →        │ (BM25+向量)   │    →     │ 上下文压缩     │
         │ 意图识别  │              │ 多路召回      │          │ 相关性过滤     │
         └──────────┘              └──────────────┘          └───────────────┘
              │                          │                         │
              └──────────────────────────┼─────────────────────────┘
                                         ▼
                                   增强生成 + 评估
                                ┌──────────────────┐
                                │ 带引用的生成       │
                                │ Faithfulness 检测 │
                                │ RAGAS 评估框架    │
                                └──────────────────┘
```

---

## 2. 严谨定义与原理

### 2.1 Query 改写（Query Rewriting）

**问题**：用户的原始查询往往不够精确或缺乏上下文，直接用于检索效果差。

**解决方案**：用 LLM 将原始查询改写为更适合检索的形式。

常见策略：

| 策略 | 做法 | 示例 |
|------|------|------|
| **扩展改写** | 补充上下文和同义词 | "RAG 怎么用" → "如何构建 Retrieval-Augmented Generation 检索增强生成系统" |
| **多查询生成** | 从不同角度生成多个查询 | 原始查询 → 3~5 个变体 → 分别检索 → 合并去重 |
| **分步分解** | 将复杂问题拆解为子问题 | "A 和 B 的区别" → "A 是什么？" + "B 是什么？" |

### 2.2 HyDE（Hypothetical Document Embeddings）

**核心洞察**（Gao et al., 2023）：查询和文档在向量空间中的分布天然不同——查询通常很短，文档较长且信息丰富。直接比较两者的 Embedding 可能不是最优的。

**HyDE 的做法**：

```
原始查询 → LLM 生成一个"假设性回答" → 对假设性回答做 Embedding → 用这个向量去检索
```

**为什么有效**：假设性回答与真实文档在形式和内容上更接近，因此在向量空间中更容易匹配到相关文档。即使假设性回答包含错误信息，其向量表示仍然能捕获正确的语义方向。

形式化地：

$$\mathbf{v}_{\text{HyDE}} = E\bigl(G(q)\bigr) \quad \text{其中 } G(q) \text{ 是 LLM 对 } q \text{ 生成的假设性回答}$$

### 2.3 Reranker（交叉编码器重排）

**问题**：Embedding 检索是"双编码器"（Bi-Encoder）架构——查询和文档独立编码，只通过向量相似度粗略比较。这牺牲了交互信息。

**解决方案**：在 Top-K 结果上，用"交叉编码器"（Cross-Encoder）精细地重排。

```
Bi-Encoder（检索阶段）:
  E(query)  →  向量 q
  E(doc)    →  向量 d     → sim(q, d)  ← 快但粗

Cross-Encoder（重排阶段）:
  [query; doc] → Encoder → score  ← 慢但精
```

**为什么交叉编码器更精确**：Cross-Encoder 将查询和文档**拼接后联合编码**，让 Transformer 的 Self-Attention 充分建模两者之间的 token 级交互。这比独立编码后只比较向量要精细得多。

**代价**：Cross-Encoder 需要对每个 (query, doc) 对独立推理，无法预计算文档向量，因此只能用于对少量候选文档（如 Top-20~50）重排。

### 2.4 混合检索（Hybrid Search）

**核心思想**：稀疏检索（BM25）和稠密检索（向量搜索）各有所长，结合两者能互补。

| 检索类型 | 擅长 | 不擅长 |
|---------|------|--------|
| **BM25（稀疏）** | 精确关键词匹配、专有名词、错误码 | 同义词、语义理解 |
| **向量检索（稠密）** | 语义理解、同义表达、跨语言 | 精确匹配、长尾关键词 |

**混合策略**：Reciprocal Rank Fusion（RRF）

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

其中 $R$ 是多个检索器的结果列表，$\text{rank}_r(d)$ 是文档 $d$ 在第 $r$ 个检索器中的排名，$k$ 是平滑常数（通常 $k=60$）。

### 2.5 Agentic RAG

**演进路线**：Naive RAG → Advanced RAG → Agentic RAG

Agentic RAG 让 **LLM Agent 控制整个检索-生成流程**，Agent 可以：
- 判断是否需要检索（简单问题直接回答）
- 选择检索哪个知识库（多数据源路由）
- 评估检索结果质量（不满意则重新检索或改写查询）
- 多轮迭代直到获得满意答案

```
用户问题 → Agent 判断
              │
              ├── 不需要检索 → 直接回答
              │
              ├── 需要检索 → 选择数据源 → 检索 → 评估结果
              │                                     │
              │                  ├── 结果足够好 → 生成答案
              │                  └── 结果不够好 → 改写 Query → 重新检索
              │
              └── 需要多步推理 → 分解子问题 → 逐步检索+推理 → 合成答案
```

### 2.6 RAG 评估框架：RAGAS

RAGAS（Retrieval Augmented Generation Assessment）提出四个核心维度：

| 维度 | 评估什么 | 计算方式 |
|------|---------|---------|
| **Faithfulness** | 答案是否忠于检索到的上下文 | LLM 将答案分解为独立声明 → 逐一检查是否被上下文支持 |
| **Answer Relevancy** | 答案是否回答了用户的问题 | 从答案反向生成问题 → 与原始问题比较相似度 |
| **Context Precision** | 检索到的上下文中相关内容排名是否靠前 | 相关文档在 Top-K 中的加权位置 |
| **Context Recall** | 回答所需的信息是否都被检索到了 | 将参考答案分解为声明 → 检查每个声明是否在上下文中出现 |

---

## 3. Python 代码实战

### 3.1 Query 改写与多查询检索

```python
"""
Query 改写：用 LLM 生成多个查询变体，扩大检索覆盖面。
依赖: pip install sentence-transformers faiss-cpu numpy
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ================================================================
# 基础设施：构建向量索引
# ================================================================

documents = [
    "检索增强生成（RAG）通过在推理时引入外部知识来提升大模型回答质量。",
    "FAISS 是一个高效的向量相似度搜索库，支持十亿级规模的近似最近邻搜索。",
    "大语言模型的幻觉问题可以通过 RAG 技术有效缓解。",
    "BM25 是一种经典的稀疏检索算法，基于词频和逆文档频率计算相关性。",
    "混合检索将 BM25 和向量检索结合，能同时利用精确匹配和语义理解。",
    "Reranker 使用交叉编码器对检索结果进行精排，显著提升排序质量。",
    "LLM 的知识截止日期问题可以通过实时检索外部数据源来解决。",
    "向量数据库如 Milvus 和 Chroma 专为高维向量存储和检索设计。",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = model.encode(documents, normalize_embeddings=True).astype(np.float32)
index = faiss.IndexFlatIP(doc_vecs.shape[1])
index.add(doc_vecs)

def search(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_vec, top_k)
    return [(documents[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


# ================================================================
# 策略 1: 简单 Query 改写（无需 LLM API 的规则化方法）
# ================================================================

def rule_based_rewrite(query: str) -> list[str]:
    """基于规则的 Query 扩展（实际中用 LLM 替代）"""
    variants = [query]  # 原始查询
    # 扩展同义表达
    synonyms = {
        "RAG": "检索增强生成 Retrieval-Augmented Generation",
        "大模型": "大语言模型 LLM",
        "向量检索": "语义搜索 向量相似度搜索",
    }
    for key, expansion in synonyms.items():
        if key in query:
            variants.append(query.replace(key, expansion))
    return variants


# ================================================================
# 策略 2: 多查询检索 + 结果合并
# ================================================================

def multi_query_retrieve(
    queries: list[str], top_k_per_query: int = 3
) -> list[tuple[str, float]]:
    """对多个查询变体分别检索，RRF 合并结果"""
    doc_scores: dict[str, float] = {}
    k = 60  # RRF 平滑常数

    for query in queries:
        results = search(query, top_k=top_k_per_query)
        for rank, (doc, _) in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc] = doc_scores.get(doc, 0) + rrf_score

    # 按 RRF 总分排序
    sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
    return sorted_docs


# ================================================================
# 对比实验
# ================================================================

original_query = "怎么解决 LLM 乱编的问题？"

# 单查询检索
print("=== 单查询检索 ===")
for rank, (doc, score) in enumerate(search(original_query, 3)):
    print(f"  [{rank+1}] ({score:.4f}) {doc[:60]}...")

# 多查询检索
rewritten = rule_based_rewrite(original_query)
rewritten.append("大语言模型幻觉问题的解决方案")  # 模拟 LLM 生成的改写
print(f"\n查询变体: {rewritten}")

print("\n=== 多查询 + RRF 检索 ===")
for rank, (doc, score) in enumerate(multi_query_retrieve(rewritten)):
    print(f"  [{rank+1}] (rrf={score:.4f}) {doc[:60]}...")
```

### 3.2 HyDE 实现

```python
"""
HyDE: 用假设性回答的 Embedding 做检索（复用 3.1 的 index 和 model）。
"""
def hyde_search(query: str, hypothetical: str, alpha: float = 0.5, top_k: int = 3):
    """alpha 控制原始查询 vs 假设回答的混合比例"""
    q_vec = model.encode([query], normalize_embeddings=True)
    h_vec = model.encode([hypothetical], normalize_embeddings=True)
    mixed = alpha * q_vec + (1 - alpha) * h_vec
    mixed = mixed / np.linalg.norm(mixed, axis=1, keepdims=True)
    scores, ids = index.search(mixed.astype(np.float32), top_k)
    return [(documents[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

# 模拟 LLM 生成的假设性回答（实际中调用 LLM API 生成）
query = "怎么评估 RAG 的效果？"
hypothetical = ("RAG 效果可从 Faithfulness、Answer Relevancy、Context Precision、"
                "Context Recall 四个维度评估，RAGAS 框架提供自动化计算。")

print("=== 标准检索 vs HyDE 检索 ===")
for label, results in [("标准", search(query, 3)),
                        ("HyDE", hyde_search(query, hypothetical))]:
    print(f"  [{label}]")
    for r, (doc, s) in enumerate(results):
        print(f"    [{r+1}] ({s:.4f}) {doc[:50]}...")
```

### 3.3 Reranker 交叉编码器重排

```python
"""
Reranker: Bi-Encoder 粗排 → Cross-Encoder 精排。
依赖: pip install sentence-transformers faiss-cpu
"""
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np, faiss

docs = [
    "RAG 通过检索外部知识增强 LLM 回答准确性。",
    "Cross-Encoder 将查询和文档拼接后联合编码，精度高但速度慢。",
    "Bi-Encoder 分别编码查询和文档，速度快但交互信息有限。",
    "Reranker 在粗排结果上做精排，是提升检索质量的关键技术。",
    "BM25 基于词频统计，适合精确关键词匹配。",
    "LangChain 提供 RAG 端到端组件化开发框架。",
]

# 阶段 1: Bi-Encoder 粗排
bi_enc = SentenceTransformer("all-MiniLM-L6-v2")
vecs = bi_enc.encode(docs, normalize_embeddings=True).astype(np.float32)
idx = faiss.IndexFlatIP(vecs.shape[1])
idx.add(vecs)

query = "检索结果重排序的方法"
q_vec = bi_enc.encode([query], normalize_embeddings=True).astype(np.float32)
bi_scores, bi_ids = idx.search(q_vec, 6)
candidates = [(docs[i], float(s)) for i, s in zip(bi_ids[0], bi_scores[0])]

# 阶段 2: Cross-Encoder 精排
cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_scores = cross_enc.predict([(query, d) for d, _ in candidates])

reranked = sorted(zip(candidates, cross_scores), key=lambda x: -x[1])
print("=== 粗排 vs 精排 ===")
for rank, ((doc, bi_s), cross_s) in enumerate(reranked):
    print(f"  [{rank+1}] cross={cross_s:.4f} bi={bi_s:.4f} | {doc[:50]}...")
```

### 3.4 混合检索（BM25 + 向量 + RRF）

```python
"""
混合检索：BM25 稀疏检索 + 向量稠密检索 + RRF 融合。
依赖: pip install rank_bm25 sentence-transformers faiss-cpu numpy
"""
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import re

documents = [
    "IndexFlatIP 是 FAISS 中的暴力搜索索引，精度 100% 但速度慢。",
    "RAG 系统需要仔细设计 Chunking 策略以保证检索质量。",
    "BM25 通过 TF-IDF 加权计算查询与文档的词汇匹配度。",
    "HNSW 索引在 FAISS 中使用层级图结构实现快速近似最近邻搜索。",
    "error code E4523 表示向量维度不匹配，请检查 Embedding 模型输出维度。",
    "向量检索通过计算 Embedding 余弦相似度找到语义最相关的文档。",
    "Reciprocal Rank Fusion (RRF) 是一种简单有效的多路结果融合算法。",
]


class HybridRetriever:
    """混合检索: BM25 + 向量 + RRF"""

    def __init__(self, docs: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs

        # 稀疏检索: BM25
        tokenized = [self._tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

        # 稠密检索: Embedding + FAISS
        self.embed_model = SentenceTransformer(model_name)
        vecs = self.embed_model.encode(docs, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs.astype(np.float32))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词（中英文混合）"""
        return re.findall(r'\w+', text.lower())

    def bm25_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_ids = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_ids]

    def vector_search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, ids = self.index.search(q_vec, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def hybrid_search(
        self, query: str, top_k: int = 5, rrf_k: int = 60
    ) -> list[tuple[str, float]]:
        """RRF 融合 BM25 和向量检索结果"""
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        vec_results = self.vector_search(query, top_k=top_k * 2)

        # RRF 计算
        rrf_scores: dict[int, float] = {}
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(vec_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return [(self.docs[doc_id], score) for doc_id, score in sorted_ids]


# ================================================================
# 对比实验
# ================================================================

retriever = HybridRetriever(documents)

# 语义查询（向量检索应占优）
q1 = "如何找到最相关的文档？"
# 精确匹配查询（BM25 应占优）
q2 = "error code E4523"

for query in [q1, q2]:
    print(f"\n查询: {query}")

    print("  [BM25]")
    for r, (idx, s) in enumerate(retriever.bm25_search(query, 3)):
        print(f"    [{r+1}] ({s:.4f}) {documents[idx][:50]}...")

    print("  [向量]")
    for r, (idx, s) in enumerate(retriever.vector_search(query, 3)):
        print(f"    [{r+1}] ({s:.4f}) {documents[idx][:50]}...")

    print("  [混合 RRF]")
    for r, (doc, s) in enumerate(retriever.hybrid_search(query, 3)):
        print(f"    [{r+1}] (rrf={s:.4f}) {doc[:50]}...")
```

### 3.5 RAGAS 评估框架使用

```python
"""
RAGAS 评估：系统性评估 RAG 质量。
依赖: pip install ragas datasets langchain langchain-openai
"""
# RAGAS 评估需要四个要素: question, answer, contexts, ground_truth

eval_data = {
    "question": ["RAG 系统的三个阶段是什么？"],
    "answer": ["RAG 包含索引构建、检索和生成三个阶段。"],
    "contexts": [["RAG 的三阶段：索引构建将文档向量化；检索阶段搜索相关文档块；生成阶段送入 LLM。"]],
    "ground_truth": ["索引构建（Indexing）、检索（Retrieval）和生成（Generation）。"],
}

# 运行 RAGAS 评估（需要 OPENAI_API_KEY）
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
#
# result = evaluate(
#     dataset=Dataset.from_dict(eval_data),
#     metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
# )
# print(result)  # {'faithfulness': 0.95, 'answer_relevancy': 0.88, ...}

# ---- 轻量级替代: 不用 API 的手动 Faithfulness 检测 ----
def manual_faithfulness(answer: str, contexts: list[str]) -> float:
    """检查答案中的声明是否被上下文支持（生产中用 LLM-as-Judge）"""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    claims = [s.strip() for s in answer.split("，") if len(s.strip()) > 5]
    if not claims:
        return 1.0
    ctx_sentences = [s.strip() for s in " ".join(contexts).split("；") if s.strip()]
    if not ctx_sentences:
        return 0.0

    claim_vecs = model.encode(claims, normalize_embeddings=True)
    ctx_vecs = model.encode(ctx_sentences, normalize_embeddings=True)
    supported = sum(1 for cv in claim_vecs if float(np.max(cv @ ctx_vecs.T)) > 0.6)
    return supported / len(claims)

print(f"Faithfulness: {manual_faithfulness(eval_data['answer'][0], eval_data['contexts'][0]):.2f}")
```

### 3.6 Agentic RAG 概念演示

```python
"""
Agentic RAG: Agent 自主决定检索策略（简化版，展示核心思想）。
依赖: pip install sentence-transformers faiss-cpu numpy
"""
from enum import Enum
from sentence_transformers import SentenceTransformer
import numpy as np, faiss

class Action(Enum):
    RETRIEVE = "retrieve"
    REWRITE = "rewrite_and_retrieve"
    DECOMPOSE = "decompose"

class AgenticRAG:
    """Agent 驱动的 RAG：路由→执行→评估→(不满意则重试)"""

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = self.model.encode(documents, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs.astype(np.float32))

    def _route(self, query: str) -> Action:
        """路由决策（生产中用 LLM 做意图识别）"""
        if len(query) < 10:
            return Action.REWRITE
        if "和" in query and "区别" in query:
            return Action.DECOMPOSE
        return Action.RETRIEVE

    def _retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q_vec, top_k)
        return [(self.documents[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def run(self, query: str, max_iter: int = 3) -> dict:
        for attempt in range(1, max_iter + 1):
            action = self._route(query)
            print(f"  [{attempt}] action={action.value}")

            if action == Action.DECOMPOSE:
                results = []
                for part in query.split("和"):
                    results.extend(self._retrieve(part.strip(), 2))
            elif action == Action.REWRITE:
                results = self._retrieve(f"{query} 详细解释 原理", 3)
            else:
                results = self._retrieve(query, 3)

            avg_score = np.mean([s for _, s in results]) if results else 0
            if avg_score >= 0.5:  # 质量阈值
                return {"contexts": [d for d, _ in results], "iterations": attempt}
            query = f"{query} 详细解释"  # 改写后重试
        return {"contexts": [], "iterations": max_iter}

# 使用示例
agent = AgenticRAG(["RAG 通过检索增强生成。", "BM25 是稀疏检索算法。",
                     "向量检索基于 Embedding 余弦相似度。", "Reranker 用交叉编码器精排。"])
for q in ["RAG", "BM25 和向量检索的区别"]:
    print(f"\n查询: {q}")
    print(f"  结果: {agent.run(q)}")
```

---

## 4. 工程师视角

### 4.1 Advanced RAG 技术选型决策

| 技术 | 何时使用 | 何时不用 | 额外成本 |
|------|---------|---------|---------|
| **Query 改写** | 用户查询模糊/口语化 | 查询已经很精确 | 1 次 LLM 调用 |
| **HyDE** | 查询与文档形式差异大 | 查询本身就像文档片段 | 1 次 LLM 调用 |
| **多查询检索** | 需要高召回率 | 低延迟要求 | N 次检索 |
| **Reranker** | 粗排结果质量不稳定 | Top-K 已经足够好 | Cross-Encoder 推理 |
| **混合检索** | 既有语义查询又有精确匹配 | 纯语义场景 | BM25 索引维护 |
| **Agentic RAG** | 复杂查询、多数据源 | 简单问答 | 多次 LLM 调用 |

### 4.2 生产级 Advanced RAG Pipeline

```
用户查询 → [1.意图路由] → [2.Query优化: 改写/HyDE/扩展]
  → [3.混合检索: BM25 + 向量 + RRF] → [4.Reranker精排 + 去重/压缩]
  → [5.Prompt组装 + LLM生成 + 引用标注] → [6.质量监控: Faithfulness + 用户反馈]
```

每个环节的技术选型在 4.1 表格中已覆盖。关键原则：**从简单开始，按需叠加**——大多数场景不需要全部六步，先跑通 3→5，再逐步加入 1/2/4/6。

### 4.3 RAG 评估的完整策略

| 评估层级 | 评估什么 | 方法 | 自动化程度 |
|---------|---------|------|-----------|
| **组件级** | Embedding 模型质量 | MTEB 排行榜 / 自建测试集 | 全自动 |
| **检索级** | 检索准确性 | Precision@K / Recall@K / MRR | 需标注数据 |
| **生成级** | 答案质量 | Faithfulness / Relevancy | LLM-as-Judge |
| **端到端** | 用户满意度 | RAGAS / 人工评估 / A/B 测试 | 半自动 |
| **生产级** | 系统稳定性 | 延迟 / 吞吐 / 错误率 | 全自动 |

### 4.4 Self-RAG 与 Corrective RAG

**Self-RAG**（Asai et al., 2023）：让模型在生成过程中自己判断：
- 是否需要检索（Retrieve token）
- 检索结果是否相关（IsRel token）
- 生成内容是否被支持（IsSup token）
- 回答是否有用（IsUse token）

**Corrective RAG**（CRAG, Yan et al., 2024）：
- 检索后用一个轻量级评估器判断结果质量
- 质量不足时触发纠正动作：改写查询 / 使用 Web 搜索 / 去除噪声文档

### 4.5 常见面试 / 系统设计问题

| 问题 | 核心要点 |
|------|---------|
| BM25 和向量检索怎么选？ | 不选，混合用；BM25 擅长精确匹配，向量擅长语义 |
| Reranker 放在哪一步？ | 粗排（Top-20~50）之后、送入 LLM 之前 |
| HyDE 的局限是什么？ | 增加一次 LLM 调用延迟；假设回答严重错误时可能误导检索 |
| 如何处理检索结果矛盾？ | 在 Prompt 中要求 LLM 识别矛盾并标注；或用多数投票 |
| RAG 评估中最重要的指标？ | Faithfulness（答案忠实于上下文）——这是 RAG 的核心价值 |
| 如何减少 RAG 系统的延迟？ | 并行化：Embedding + 检索并行；缓存高频查询；流式生成 |

---

## 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| **Query 改写** | 用 LLM 将模糊查询改写为检索友好形式 |
| **HyDE** | 用假设性回答的 Embedding 替代查询向量做检索 |
| **Reranker** | Cross-Encoder 对粗排结果精排，精度高但只能处理少量候选 |
| **混合检索** | BM25 + 向量检索 + RRF 融合，精确匹配与语义理解互补 |
| **Agentic RAG** | Agent 自主决策：是否检索、检索什么、结果是否合格 |
| **RAGAS** | 四维度评估框架：Faithfulness / Relevancy / Precision / Recall |
| **Self-RAG** | 模型在生成中自判断是否需要检索和检索质量 |

**下一节**：[论文与FAQ](论文与FAQ.md)——系统梳理 RAG 领域的关键论文和工程实践中的常见困惑。
