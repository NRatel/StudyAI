# RAG 完整流程

> **前置知识**：[预训练语言模型演进](../AI学习_04_预训练语言模型演进/README.md)（理解 Embedding、Encoder/Decoder 架构）
>
> **本节目标**：理解 RAG 解决什么问题、三阶段流程（索引构建→检索→生成）每一步做什么、Naive RAG 与 Advanced RAG 的本质区别。

---

## 1. 直觉与概述

### 1.1 核心问题：大模型为什么需要"外挂记忆"？

ChatGPT 能流畅回答各种问题，但它有三个致命短板：

| 问题 | 表现 | 根因 |
|------|------|------|
| **幻觉（Hallucination）** | 一本正经地编造不存在的论文、API、数据 | 模型是概率分布采样器，不是知识数据库 |
| **时效性缺失** | 无法回答训练截止日期之后发生的事 | 参数在训练后就"冻结"了 |
| **无法访问私有数据** | 回答不了公司内部文档的问题 | 预训练语料中没有这些内容 |

**RAG 的核心思路**：不让模型"记住"所有知识，而是在生成前先从外部知识库**检索**相关信息，把检索结果作为**上下文**喂给模型，让它"有据可依"地回答。

类比：**RAG = 开卷考试**。你不需要记住所有内容，只要能快速翻到正确的参考资料，就能写出高质量答案。

### 1.2 RAG 三阶段总览

```
┌───────────────────────────────────────────────────────┐
│                 离线阶段（Indexing）                    │
│                                                       │
│  原始文档 → 解析 → 分块(Chunking) → Embedding → 向量库 │
└──────────────────────┬────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────┐
│              在线阶段（Retrieval + Generation）         │
│                                                       │
│  用户问题                                              │
│    │                                                  │
│    ├─→ Embedding ─→ 向量检索 ─→ Top-K 相关文档         │
│    │                                  │               │
│    └────────────────┐                 │               │
│                     ▼                 ▼               │
│              ┌─────────────────────────┐              │
│              │  Prompt = 问题 + 文档块  │              │
│              │         ↓               │              │
│              │     LLM 生成最终答案     │              │
│              └─────────────────────────┘              │
└───────────────────────────────────────────────────────┘
```

三阶段拆解：

| 阶段 | 核心动作 | 关键组件 |
|------|---------|---------|
| **索引构建（Indexing）** | 文档分块 → 向量化 → 存入向量数据库 | Chunker + Embedding Model + Vector DB |
| **检索（Retrieval）** | 用户查询向量化 → 相似度搜索 → 返回 Top-K | Embedding Model + Vector DB + (Reranker) |
| **生成（Generation）** | 拼接检索文档与用户问题 → 送入 LLM | LLM + Prompt Template |

---

## 2. 严谨定义与原理

### 2.1 形式化定义

给定：
- 知识库 $\mathcal{D} = \{d_1, d_2, \ldots, d_N\}$（$N$ 个文档块）
- 用户查询 $q$
- Embedding 函数 $E: \text{text} \to \mathbb{R}^d$
- 生成模型 $G$（即 LLM）

RAG 的过程分解为三步：

**Step 1 — 索引构建**（离线）：

$$\mathbf{v}_i = E(d_i), \quad i = 1, \ldots, N$$

将每个文档块编码为 $d$ 维向量，存入向量数据库。

**Step 2 — 检索**（在线）：

$$\mathbf{v}_q = E(q)$$

$$\mathcal{R} = \text{Top-K}_{d_i \in \mathcal{D}} \bigl[\text{sim}(\mathbf{v}_q,\; \mathbf{v}_i)\bigr]$$

计算查询与所有文档块的相似度，返回最相关的 $K$ 个。

**Step 3 — 生成**（在线）：

$$\text{answer} = G\bigl(\text{prompt}(q,\; \mathcal{R})\bigr)$$

将查询与检索到的文档块拼装为 Prompt，送入 LLM 生成答案。

### 2.2 为什么 RAG 有效？

LLM 的参数量是有限的，无法无损压缩人类所有知识。RAG 本质上是在推理时动态引入**外部信息源**，使得：

$$P(\text{answer} \mid q, \mathcal{R}) \gg P(\text{answer} \mid q)$$

有了检索到的相关文档后，生成正确答案的概率显著提升。Lewis et al. (2020) 在原始 RAG 论文中实验性地证明了这一点：在开放域问答任务上，RAG 模型在 Natural Questions 和 TriviaQA 上的 Exact Match 分数显著超过纯生成模型。

### 2.3 RAG vs 其他方案

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **RAG** | 无需重训、知识可实时更新、答案可溯源 | 检索质量是瓶颈、有检索延迟 | 知识密集型问答、企业知识库 |
| **微调** | 模型内化领域知识、无检索延迟 | 训练成本高、无法实时更新 | 固定领域、风格适配 |
| **长上下文** | 实现简单，把文档全塞进 Context | 成本随 token 数线性增长 | 小规模文档（< 100K tokens） |
| **知识图谱** | 结构化推理、精确实体关系查询 | 构建成本高、覆盖面有限 | 金融/医疗/法律的实体关系推理 |

**关键洞察**：RAG 和微调不是二选一——最佳实践往往是 **RAG + 微调** 结合。先 RAG 获取外部知识，用微调过的模型更好地理解和组织检索内容。

### 2.4 Naive RAG vs Advanced RAG vs Modular RAG

```
Naive RAG（2020~2022）
  │  简单的 "检索 + 拼接 + 生成"
  │  问题：检索质量参差、对复杂查询无力、生成受噪声干扰
  ▼
Advanced RAG（2023~）
  │  在三阶段的每一步都做优化
  │  预检索：Query 改写、HyDE
  │  检索中：混合检索（稀疏 + 稠密）
  │  后检索：Reranker 重排、上下文压缩
  ▼
Modular RAG（2024~）
  │  将 RAG 拆解为可插拔模块，按需组合
  │  Agentic RAG、Graph RAG、Adaptive RAG、Self-RAG
```

| 维度 | Naive RAG | Advanced RAG | Modular RAG |
|------|-----------|-------------|-------------|
| **检索前** | 直接用原始查询 | Query 改写 / HyDE | Agent 决定是否检索、用何策略 |
| **检索** | 单一向量相似度 | 混合检索（BM25 + 向量） | 多轮检索、知识图谱增强 |
| **检索后** | 直接拼接到 Prompt | Reranker 重排、上下文压缩 | 相关性过滤 + 验证 + 自反思 |
| **生成** | 单次生成 | 带引用/置信度的生成 | 迭代生成、自我纠错 |

---

## 3. Python 代码实战

### 3.1 从零构建最小 RAG（纯 Python + FAISS）

```python
"""
最小 RAG 系统：不依赖上层框架，理解每一步在做什么。
依赖: pip install faiss-cpu sentence-transformers
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ================================================================
# 阶段 1：索引构建（Indexing）—— 离线执行一次
# ================================================================

# 模拟企业知识库文档
documents = [
    "RAG（Retrieval-Augmented Generation）由 Facebook AI Research 在 2020 年提出，"
    "核心思想是生成答案前先从外部知识库检索相关文档。",

    "FAISS 是 Meta 开源的向量相似度搜索库，支持十亿级向量的高效检索，"
    "核心算法包括 IVF（倒排文件索引）和 HNSW（层级可导航小世界图）。",

    "LangChain 是构建 LLM 应用的开源框架，提供 Document Loader、"
    "Text Splitter、Embedding、Vector Store、Chain 等模块化组件。",

    "向量数据库（Milvus / Chroma / Pinecone）专为高维向量存储和检索设计，"
    "支持 ANN 近似最近邻搜索，在毫秒级返回相似结果。",

    "Embedding 模型将文本映射为稠密向量表示。常用模型有 OpenAI text-embedding-3-small、"
    "BGE、GTE、E5 等，输出维度通常为 384~3072。",

    "Chunking（文档分块）是 RAG 的关键预处理环节。常见策略包括固定大小分块、"
    "按段落分块、递归字符分块、基于语义相似度的分块。",
]

# 加载 Embedding 模型（本地运行，无需 API）
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 维，约 80MB
doc_embeddings = embed_model.encode(documents, normalize_embeddings=True)
print(f"文档数量: {len(documents)}, 向量维度: {doc_embeddings.shape}")
# 输出: 文档数量: 6, 向量维度: (6, 384)

# 构建 FAISS 索引（L2 归一化后用内积 = 余弦相似度）
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings.astype(np.float32))
print(f"FAISS 索引向量数: {index.ntotal}")

# ================================================================
# 阶段 2：检索（Retrieval）—— 每次查询在线执行
# ================================================================

def retrieve(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    """编码查询 → FAISS 搜索 → 返回 (文档, 分数) 列表"""
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_vec.astype(np.float32), top_k)
    return [
        (documents[idx], float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx != -1
    ]

query = "RAG 是什么？它是如何工作的？"
results = retrieve(query, top_k=3)
print(f"\n查询: {query}")
for rank, (doc, score) in enumerate(results, 1):
    print(f"  [{rank}] (score={score:.4f}) {doc[:60]}...")

# ================================================================
# 阶段 3：生成（Generation）—— 拼接 Prompt 后调用 LLM
# ================================================================

def build_rag_prompt(question: str, contexts: list[str]) -> str:
    """将问题与检索到的上下文组装为 Prompt"""
    ctx_block = "\n\n".join(
        f"[参考文档 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return f"""你是一个专业的 AI 技术助手。请严格根据以下参考文档回答用户问题。
如果参考文档中没有相关信息，请明确说明"根据现有资料无法回答"。
请在回答中标注信息来源（如 [参考文档 1]）。

{ctx_block}

用户问题：{question}

回答："""

# 组装 Prompt
contexts = [doc for doc, _ in results]
prompt = build_rag_prompt(query, contexts)
print(f"\n=== 生成的 Prompt（前 400 字符）===\n{prompt[:400]}...")

# 调用 LLM（取消注释后可实际运行）
# from openai import OpenAI
# client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": prompt}],
#     temperature=0.2,  # 低温度 → 更忠实于上下文
# )
# print(response.choices[0].message.content)
```

### 3.2 使用 LangChain 构建 RAG Pipeline

```python
"""
LangChain RAG Pipeline：生产级组件化方案。
依赖: pip install langchain langchain-community faiss-cpu sentence-transformers
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================================================================
# Step 1: 文档分块
# ================================================================

raw_text = """
Retrieval-Augmented Generation (RAG) 将信息检索与文本生成结合。系统首先将知识库
中的文档分块并向量化，存储到向量数据库。当用户提出问题时，系统将问题转为向量，
从向量数据库检索最相关的文档块，然后将检索结果作为上下文送入大语言模型生成答案。

RAG 的核心优势在于：
1. 减少幻觉——模型回答有据可依
2. 知识可更新——只需更新知识库，不必重新训练模型
3. 答案可溯源——能追溯到具体的源文档段落
4. 成本可控——不需要微调大模型

向量数据库的选择取决于规模和运维要求：FAISS 适合嵌入式场景，Milvus 适合分布式
生产环境，Chroma 适合快速原型，Pinecone 提供全托管服务。
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块最大 200 字符
    chunk_overlap=30,     # 块间重叠 30 字符
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
chunks = splitter.split_text(raw_text)
print(f"分块结果: {len(chunks)} 块")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: ({len(c)} chars) {c[:50]}...")

# ================================================================
# Step 2: 向量化 + 构建索引
# ================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)
vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
print(f"\n索引构建完成，向量数: {vectorstore.index.ntotal}")

# ================================================================
# Step 3: 检索
# ================================================================

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

query = "RAG 有什么优势？"
docs = retriever.invoke(query)
print(f"\n查询: {query}")
for i, doc in enumerate(docs):
    print(f"  [{i+1}] {doc.page_content[:80]}...")

# ================================================================
# Step 4: 完整 Chain（需配置 LLM API Key）
# ================================================================
# from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
#
# template = PromptTemplate(
#     template="根据上下文回答问题。\n\n上下文：\n{context}\n\n问题：{question}\n回答：",
#     input_variables=["context", "question"],
# )
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# chain = RetrievalQA.from_chain_type(
#     llm=llm, retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": template},
#     return_source_documents=True,
# )
# result = chain.invoke({"query": "RAG 有什么优势？"})
# print(f"答案: {result['result']}")
```

### 3.3 Naive RAG 完整实现（面向工程的类封装）

```python
"""
NaiveRAG 类：封装完整的 Naive RAG 流程。
目的是暴露 Naive RAG 的局限性，为 Advanced RAG 做铺垫。
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    """一个文档块"""
    content: str
    source: str = ""
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class NaiveRAG:
    """
    Naive RAG 的完整实现。

    已知局限（Advanced RAG 逐一解决这些问题）：
    1. 固定窗口分块 → 可能切断完整语义
    2. 单一稠密向量检索 → 可能漏掉精确关键词匹配
    3. 无 Query 改写 → 用户问题模糊时检索效果差
    4. 无 Reranker → Top-K 排序不够精确
    5. 无上下文压缩 → 噪声文档干扰生成
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.dim = self.embed_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: list[Chunk] = []

    # ---------- 索引构建 ----------

    def _fixed_window_chunk(
        self, text: str, size: int = 300, overlap: int = 50
    ) -> list[str]:
        """固定窗口 + 滑动重叠分块"""
        pieces = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            pieces.append(text[start:end])
            start += size - overlap
        return pieces

    def ingest(self, texts: list[str], sources: list[str] | None = None):
        """批量入库: 分块 → 编码 → 写入 FAISS"""
        sources = sources or ["unknown"] * len(texts)
        new_chunks = []
        for text, src in zip(texts, sources):
            for piece in self._fixed_window_chunk(text):
                new_chunks.append(Chunk(content=piece, source=src))

        vecs = self.embed_model.encode(
            [c.content for c in new_chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.index.add(vecs.astype(np.float32))
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, ids = self.index.search(q_vec, top_k)
        return [
            (self.chunks[i], float(s))
            for i, s in zip(ids[0], scores[0]) if i != -1
        ]

    # ---------- 提示词组装 ----------

    def build_prompt(self, query: str, top_k: int = 5) -> str:
        hits = self.retrieve(query, top_k)
        ctx = "\n\n".join(
            f"[{r+1}] (来源: {c.source})\n{c.content}"
            for r, (c, _) in enumerate(hits)
        )
        return (
            f"根据以下参考资料回答用户问题。如果资料不足请说明。\n\n"
            f"参考资料：\n{ctx}\n\n"
            f"用户问题：{query}\n"
            f"回答："
        )


# ---------- 运行示例 ----------
if __name__ == "__main__":
    rag = NaiveRAG()
    n = rag.ingest(
        texts=[
            "FAISS 支持多种索引: IndexFlatL2（暴力搜索）、IndexIVFFlat（倒排索引）、"
            "IndexHNSW（图索引）。百万级向量推荐 IVF+PQ 组合。",
            "LangChain 的核心 RAG 组件：Document Loader → Text Splitter → "
            "Embedding → Vector Store → Retriever → Chain。",
        ],
        sources=["faiss_docs", "langchain_docs"],
    )
    print(f"入库 {n} 个 chunk，总计 {len(rag.chunks)}")
    print(rag.build_prompt("FAISS 有哪些索引类型？", top_k=3))
```

---

## 4. 工程师视角

### 4.1 RAG 系统的关键设计决策

搭建 RAG 系统时，需要做出以下核心决策（每个决策在后续章节详细展开）：

| 决策 | 选项 | 影响 |
|------|------|------|
| **Chunking 策略** | 固定窗口 / 递归字符 / 语义分块 | 直接决定检索粒度和质量 |
| **Embedding 模型** | 开源（BGE/GTE）/ 闭源（OpenAI/Cohere） | 决定语义表示质量 |
| **向量数据库** | FAISS / Chroma / Milvus / Pinecone | 决定规模上限和运维成本 |
| **检索策略** | 纯向量 / BM25 / 混合 / Reranker | 检索精度的核心 |
| **Top-K 选择** | 3~10 | 太少信息不足，太多引入噪声 |
| **LLM 选择** | GPT-4o / Claude / 本地模型 | 成本与质量的权衡 |
| **Prompt 设计** | 角色设定 / 引用格式 / 拒答策略 | 生成质量和可控性 |

### 4.2 生产环境 RAG 架构

```
             ┌───────────┐
             │  用户请求   │
             └─────┬─────┘
                   │
            ┌──────▼──────┐
            │  API Gateway │  ← 限流、鉴权、日志
            └──────┬──────┘
                   │
        ┌──────────▼──────────┐
        │   RAG Orchestrator   │  ← FastAPI / LangServe
        │  ┌────────────────┐  │
        │  │ Query 预处理    │  │  ← 改写、HyDE、意图识别
        │  │ 检索 + Rerank  │  │  ← 混合检索 + 重排
        │  │ Prompt 组装    │  │  ← 模板 + 上下文压缩
        │  │ LLM 调用      │  │  ← 流式输出
        │  │ 后处理        │  │  ← 引用标注、安全过滤
        │  └────────────────┘  │
        └──┬──────┬──────┬─────┘
           │      │      │
     ┌─────▼──┐ ┌─▼───┐ ┌▼──────┐
     │Embedding│ │向量库│ │ LLM   │
     │Service  │ │     │ │ API   │
     └────────┘ └─────┘ └───────┘

离线数据管线：
  文档源(S3/DB) → ETL(解析) → Chunker → Embedding → Vector DB
                                         ↑
                              定时增量更新 ─┘
```

### 4.3 典型失败模式与对策

| 失败模式 | 表现 | 根因 | 解决方案 |
|---------|------|------|---------|
| 检索空 | 回答"不知道"或幻觉 | Embedding 语义鸿沟 | Query 改写、HyDE、调整 Chunk 策略 |
| 检索到无关文档 | 答案跑题 | 向量相似 ≠ 语义相关 | 增加 Reranker、混合检索 |
| 检索到但答案错 | LLM 忽略或曲解上下文 | Prompt 设计不佳 / LLM 能力不足 | 优化 Prompt、换更强模型 |
| 延迟过高 | 用户体验差 | Embedding 计算慢 / 索引未优化 | ANN 索引、Embedding 缓存、流式输出 |
| 知识过时 | 答案过时 | 知识库未更新 | 建立定期/事件驱动更新管线 |
| 数据泄露 | 返回了用户无权访问的内容 | 未做权限过滤 | 元数据权限标记 + 检索时过滤 |

### 4.4 RAG 与微调的决策矩阵

| 需求 | 推荐方案 | 原因 |
|------|---------|------|
| 知识经常更新 | RAG | 改知识库即可，无需重训 |
| 需要答案溯源 | RAG | 天然支持引用标注 |
| 需要特定输出风格 | 微调 | 风格是模型行为，非知识 |
| 极低延迟要求 | 微调 | 省去检索环节 |
| 两者都需要 | RAG + 微调 | 微调模型 + RAG 检索是最佳实践 |

### 4.5 常见面试 / 系统设计问题

| 问题 | 核心要点 |
|------|---------|
| RAG 和 Fine-tuning 怎么选？ | RAG 用于知识注入（可更新、可溯源）；Fine-tuning 用于行为适配（格式/风格）；可结合 |
| chunk_size 怎么定？ | 起点 256~512 tokens；太小丢上下文，太大引入噪声；需对比实验 |
| RAG 系统如何保证安全？ | 元数据权限标记 → 检索时过滤 → 输出安全审查 → 审计日志 |
| Top-K 设多少？ | 通常 3~5；多了会稀释有用信息并增加 token 成本 |
| 如何处理多模态文档？ | 图片用 VLM 生成描述文字 → 和文本一起索引；或用多模态 Embedding |

---

## 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| **RAG** | 生成前先检索外部知识，让 LLM "有据可依"地回答 |
| **三阶段** | 索引构建（离线）→ 检索（在线）→ 生成（在线） |
| **为什么需要 RAG** | 解决 LLM 的幻觉、时效性缺失、私有数据隔离三大硬伤 |
| **Naive RAG** | 最简实现：分块→编码→检索→拼接→生成，有效但粗糙 |
| **Advanced RAG** | 在每一步增加优化：Query 改写、混合检索、Reranker |
| **Modular RAG** | 模块化可插拔架构，支持 Agent 驱动的自适应检索 |
| **核心设计决策** | Chunking 策略、Embedding 模型、向量数据库、检索策略 |

**下一节**：深入 Embedding 与向量检索——RAG 系统的"心脏"如何将文本变为可搜索的向量？
