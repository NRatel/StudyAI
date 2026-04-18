# Embedding 与向量检索

> **前置知识**：[RAG 完整流程](1_RAG完整流程.md)（已理解 RAG 三阶段、索引构建的基本流程）
>
> **本节目标**：深入 RAG 的核心引擎——Embedding 模型如何工作、向量数据库如何存储和检索、相似度搜索的算法原理、Chunking 策略如何影响检索质量。

---

## 1. 直觉与概述

### 1.1 Embedding 是什么？

Embedding 是将离散的文本（词、句子、段落）映射为连续的高维向量的过程。在 RAG 场景中，核心需求是**句子级 / 段落级 Embedding**——将一整段话压缩为一个向量，使得语义相近的文本在向量空间中距离相近。

```
"如何搭建一个 RAG 系统？"  →  [0.12, -0.34, 0.56, ..., 0.78]  ← 384~3072 维
"RAG 系统的构建步骤"       →  [0.11, -0.33, 0.55, ..., 0.77]  ← 两者很接近！
"今天天气不错"             →  [-0.45, 0.67, -0.12, ..., 0.23] ← 差异很大
```

**关键区别**：
- **词级 Embedding**（Word2Vec/GloVe）：每个词一个向量，无法捕获句子整体语义
- **句子级 Embedding**（Sentence-BERT/BGE/GTE）：整个句子/段落一个向量，专门针对语义相似度优化
- **Token 级 Embedding**（BERT 最后一层）：每个 token 一个向量，适合 NER/分类，不适合直接做检索

### 1.2 向量检索的核心挑战

当知识库有百万甚至十亿级文档块时，对每个查询都做暴力搜索（逐一计算相似度）是不现实的。需要高效的近似最近邻（ANN）算法。

```
暴力搜索: O(N * d)  ← N=10亿，d=1024 → 每次查询数十秒
ANN 搜索: O(log N)  ← 毫秒级返回，精度损失 < 5%
```

### 1.3 本节知识地图

```
Embedding 模型            向量数据库              Chunking 策略
  │                        │                      │
  ├── 训练目标（对比学习）   ├── 索引结构            ├── 固定窗口
  ├── 模型选型             │   ├── Flat（暴力）     ├── 递归字符
  └── 推理优化             │   ├── IVF（倒排）      ├── 语义分块
                           │   ├── HNSW（图）       └── 文档结构感知
                           │   └── PQ（量化）
                           └── 选型（FAISS/Milvus/Chroma）
```

---

## 2. 严谨定义与原理

### 2.1 句子 Embedding 的训练

现代句子 Embedding 模型（如 Sentence-BERT、BGE、GTE）的训练通常基于**对比学习（Contrastive Learning）**。

**核心思想**：给定一个查询 $q$，有一个正例文档 $d^+$（与 $q$ 语义相关）和多个负例文档 $d^-_1, \ldots, d^-_n$（不相关）。训练目标是让 $E(q)$ 与 $E(d^+)$ 的相似度尽可能高，与 $E(d^-_i)$ 的相似度尽可能低。

**InfoNCE 损失函数**：

$$\mathcal{L} = -\log \frac{\exp\bigl(\text{sim}(E(q), E(d^+)) / \tau\bigr)}{\exp\bigl(\text{sim}(E(q), E(d^+)) / \tau\bigr) + \sum_{i=1}^{n} \exp\bigl(\text{sim}(E(q), E(d^-_i)) / \tau\bigr)}$$

其中 $\tau$ 是温度超参数，$\text{sim}$ 通常是余弦相似度。

**Sentence-BERT 的做法**：

```
输入句子 A  →  BERT Encoder  →  [CLS] 或 Mean Pooling  →  向量 u
输入句子 B  →  BERT Encoder  →  [CLS] 或 Mean Pooling  →  向量 v
                                                              │
                              cosine_similarity(u, v) ────────┘
```

关键改进：BERT 原生的 [CLS] 向量**不适合**直接做句子相似度（因为没有针对相似度目标训练）。Sentence-BERT 通过在 NLI（自然语言推理）和 STS（语义文本相似度）数据上微调，让输出向量真正反映语义相似度。

### 2.2 相似度度量

| 度量 | 公式 | 取值范围 | 特点 |
|------|------|---------|------|
| **余弦相似度** | $\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | $[-1, 1]$ | 只关注方向，不关注长度；最常用 |
| **内积（点积）** | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$ | $(-\infty, +\infty)$ | 向量归一化后等价于余弦；FAISS 默认用 |
| **欧氏距离** | $\|\mathbf{u} - \mathbf{v}\|_2$ | $[0, +\infty)$ | 越小越相似；对绝对值敏感 |

**实用选择**：先将向量 L2 归一化，然后用**内积**搜索。归一化后内积 = 余弦相似度，且内积计算更快。

### 2.3 近似最近邻（ANN）算法

#### IVF（Inverted File Index）

```
1. 将向量空间用 K-Means 聚类为 nlist 个簇
2. 每个向量被分配到最近的簇中
3. 查询时，先找到 nprobe 个最近的簇
4. 只在这些簇内做精确搜索

时间复杂度: O(nlist + nprobe * N/nlist * d)
典型参数: nlist=1024, nprobe=16 → 只搜索 1.6% 的数据
```

#### HNSW（Hierarchical Navigable Small World）

```
1. 构建多层图结构，上层稀疏（长程连接），下层稠密（短程连接）
2. 搜索从最上层开始，贪心寻找最近邻
3. 逐层下降，在每层精细化搜索

优点: 查询速度极快（毫秒级），召回率高（> 95%）
缺点: 内存占用大（需存储图结构），构建时间长
```

#### PQ（Product Quantization）

```
1. 将 d 维向量切分为 m 个子空间
2. 每个子空间独立做 K-Means 聚类（通常 K=256，即 8 bit）
3. 每个向量只存 m 个聚类 ID（m 字节），而非 d 个 float（4d 字节）

压缩比: d*4 / m ≈ 100~400 倍
代价: 一定的精度损失（通常可接受）
```

### 2.4 Chunking（文档分块）策略

分块质量直接决定检索质量。核心权衡：**粒度 vs 完整性**。

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **固定窗口** | 每 N 个字符/token 切一刀，加 overlap | 简单快速 | 可能切断句子/段落 |
| **递归字符** | 按 `\n\n` → `\n` → `。` → ` ` 递归切分 | 尊重文本结构 | 块大小不均匀 |
| **语义分块** | 计算相邻句子的 Embedding 相似度，低于阈值则切分 | 语义完整性好 | 计算成本高 |
| **文档结构感知** | 按 Markdown 标题 / HTML 标签 / PDF 章节切分 | 对结构化文档效果最好 | 需要文档解析器 |
| **Parent-Child** | 小块用于检索，大块用于送入 LLM | 检索精度高 + 上下文完整 | 实现略复杂 |

**经验法则**：
- 通用场景：RecursiveCharacterTextSplitter，chunk_size=512，overlap=50
- 技术文档：按标题/章节结构分块
- 对话记录：按轮次分块
- 代码文件：按函数/类分块

---

## 3. Python 代码实战

### 3.1 Embedding 模型对比实验

```python
"""
对比不同 Embedding 模型在语义检索上的表现。
依赖: pip install sentence-transformers numpy
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# 准备测试数据
queries = [
    "如何搭建 RAG 系统？",
    "Python 怎么读取 JSON 文件？",
]

documents = [
    "RAG 系统的构建步骤包括文档分块、向量化、检索和生成。",
    "使用 Python 的 json 模块可以方便地读写 JSON 格式的数据。",
    "今天的天气预报显示明天会下雨。",
    "检索增强生成技术将外部知识库与大语言模型结合。",
    "import json; data = json.load(open('file.json')) 即可读取 JSON 文件。",
]

# 加载两个不同的模型进行对比
models = {
    "MiniLM-L6": SentenceTransformer("all-MiniLM-L6-v2"),       # 轻量级
    "MiniLM-L12": SentenceTransformer("all-MiniLM-L12-v2"),     # 稍大
}

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """批量计算余弦相似度"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T

for name, model in models.items():
    print(f"\n=== 模型: {name} (维度: {model.get_sentence_embedding_dimension()}) ===")
    q_vecs = model.encode(queries)
    d_vecs = model.encode(documents)
    sim_matrix = cosine_sim(q_vecs, d_vecs)

    for i, query in enumerate(queries):
        print(f"\n  查询: {query}")
        ranked = np.argsort(-sim_matrix[i])
        for rank, j in enumerate(ranked):
            print(f"    [{rank+1}] ({sim_matrix[i][j]:.4f}) {documents[j][:50]}...")
```

### 3.2 FAISS 索引类型对比

```python
"""
FAISS 不同索引类型的性能与精度对比。
依赖: pip install faiss-cpu numpy sentence-transformers
"""
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 生成模拟数据（实际场景中来自真实文档的 Embedding）
np.random.seed(42)
dim = 384
n_vectors = 100_000  # 10 万向量
n_queries = 100

# 用随机向量模拟（实际中用 Embedding 模型编码文档）
database = np.random.randn(n_vectors, dim).astype(np.float32)
faiss.normalize_L2(database)  # L2 归一化

queries = np.random.randn(n_queries, dim).astype(np.float32)
faiss.normalize_L2(queries)

def benchmark(index, name: str, queries: np.ndarray, top_k: int = 10):
    """评估索引的搜索速度和召回率"""
    start = time.perf_counter()
    scores, indices = index.search(queries, top_k)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "time_ms": elapsed * 1000,
        "qps": len(queries) / elapsed,
    }

# ---- 1. Flat（暴力搜索，作为基准） ----
flat_index = faiss.IndexFlatIP(dim)
flat_index.add(database)
gt_scores, gt_indices = flat_index.search(queries, 10)  # 地面真值

# ---- 2. IVF ----
nlist = 256
ivf_index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist)
ivf_index.train(database)
ivf_index.add(database)
ivf_index.nprobe = 16

# ---- 3. HNSW ----
hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # M=32
hnsw_index.hnsw.efSearch = 64
hnsw_index.add(database)

# ---- 4. IVF + PQ ----
m = 48  # 子空间数量
ivfpq_index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, m, 8)
ivfpq_index.train(database)
ivfpq_index.add(database)
ivfpq_index.nprobe = 16

# 运行基准测试
results = []
for idx, name in [
    (flat_index, "Flat (暴力)"),
    (ivf_index, "IVF"),
    (hnsw_index, "HNSW"),
    (ivfpq_index, "IVF+PQ"),
]:
    r = benchmark(idx, name, queries)

    # 计算召回率 (与 Flat 对比)
    _, pred_indices = idx.search(queries, 10)
    recall = np.mean([
        len(set(gt_indices[i]) & set(pred_indices[i])) / 10
        for i in range(len(queries))
    ])
    r["recall@10"] = recall
    results.append(r)

print(f"\n{'索引类型':<15} {'耗时(ms)':<12} {'QPS':<10} {'Recall@10':<10}")
print("-" * 50)
for r in results:
    print(f"{r['name']:<15} {r['time_ms']:<12.2f} {r['qps']:<10.0f} {r['recall@10']:<10.4f}")
```

### 3.3 Chunking 策略对比

```python
"""
对比不同分块策略的效果差异。
依赖: pip install langchain sentence-transformers numpy
"""
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import numpy as np
from sentence_transformers import SentenceTransformer

# 模拟一篇技术文档
document = """# RAG 系统架构

## 1. 索引构建

索引构建是 RAG 的离线阶段。首先需要将原始文档进行解析，支持 PDF、HTML、Markdown 等格式。解析后的文本需要进行分块处理，常见的分块策略包括固定窗口分块和语义分块。

分块后的文本通过 Embedding 模型转换为向量表示。常用的 Embedding 模型包括 BGE、GTE、E5 等开源模型，以及 OpenAI 的 text-embedding-3-small 等商业 API。

生成的向量存储到向量数据库中，如 FAISS、Milvus、Chroma 等。

## 2. 在线检索

当用户发起查询时，系统将查询文本通过相同的 Embedding 模型转换为向量。然后在向量数据库中执行相似度搜索，返回 Top-K 个最相关的文档块。

相似度度量通常使用余弦相似度或内积。为了提高检索质量，可以使用混合检索策略，将稀疏检索（如 BM25）与稠密向量检索结合。

## 3. 增强生成

检索到的文档块与用户查询一起组装为 Prompt，送入大语言模型生成最终答案。Prompt 设计需要明确指示模型基于给定上下文回答，避免使用自身知识编造内容。"""

# ---- 策略 1: 固定字符分块 ----
fixed_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
)
fixed_chunks = fixed_splitter.split_text(document)

# ---- 策略 2: 递归字符分块（LangChain 推荐默认方案）----
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(document)

# ---- 策略 3: Markdown 标题感知分块 ----
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
    ]
)
md_chunks_raw = md_splitter.split_text(document)
md_chunks = [doc.page_content for doc in md_chunks_raw]

# ---- 对比输出 ----
for name, chunks in [
    ("固定字符", fixed_chunks),
    ("递归字符", recursive_chunks),
    ("Markdown 感知", md_chunks),
]:
    print(f"\n=== {name} ({len(chunks)} 块) ===")
    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)} chars) {c[:60]}...")

# ---- 用 Embedding 验证语义完整性 ----
model = SentenceTransformer("all-MiniLM-L6-v2")
query = "RAG 系统如何进行在线检索？"
q_vec = model.encode([query], normalize_embeddings=True)

print(f"\n=== 语义检索对比 (查询: '{query}') ===")
for name, chunks in [
    ("固定字符", fixed_chunks),
    ("递归字符", recursive_chunks),
    ("Markdown 感知", md_chunks),
]:
    c_vecs = model.encode(chunks, normalize_embeddings=True)
    sims = (q_vec @ c_vecs.T)[0]
    best_idx = np.argmax(sims)
    print(f"\n  [{name}] 最相关块 (score={sims[best_idx]:.4f}):")
    print(f"    {chunks[best_idx][:100]}...")
```

### 3.4 向量数据库实战：FAISS 持久化 + Chroma

```python
"""
向量数据库持久化与多数据库对比。
依赖: pip install faiss-cpu chromadb sentence-transformers
"""
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "FAISS 支持多种索引类型，包括暴力搜索、IVF 和 HNSW。",
    "Chroma 是一个轻量级向量数据库，适合快速原型。",
    "Milvus 是分布式向量数据库，适合十亿级生产环境。",
    "Embedding 模型将文本转换为稠密向量，是 RAG 核心。",
]
vectors = model.encode(documents, normalize_embeddings=True).astype(np.float32)

# ---- FAISS: 保存与加载 ----
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "/tmp/rag_demo.index")

loaded = faiss.read_index("/tmp/rag_demo.index")
q = model.encode(["向量检索用什么数据库？"], normalize_embeddings=True).astype(np.float32)
scores, ids = loaded.search(q, 3)
for i, (idx, s) in enumerate(zip(ids[0], scores[0])):
    print(f"  [{i+1}] ({s:.4f}) {documents[idx]}")

# ---- Chroma: 嵌入式向量数据库 ----
import chromadb
client = chromadb.PersistentClient(path="/tmp/chroma_rag_demo")
collection = client.get_or_create_collection("rag_docs", metadata={"hnsw:space": "cosine"})
collection.add(documents=documents, ids=[f"doc_{i}" for i in range(len(documents))])

results = collection.query(query_texts=["向量数据库有哪些？"], n_results=3)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  ({dist:.4f}) {doc[:60]}...")
```

### 3.5 语义分块实现

```python
"""
基于 Embedding 相似度的语义分块。
当相邻句子的语义相似度低于阈值时，在此处切分。
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer


def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.5,
    min_chunk_size: int = 50,
) -> list[str]:
    """
    语义分块算法：
    1. 先按句子拆分
    2. 计算相邻句子的 Embedding 余弦相似度
    3. 相似度低于阈值的位置作为切分点
    4. 合并切分后过小的块
    """
    # 按句子拆分
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # 编码所有句子
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(embeddings[i] @ embeddings[i + 1])
        similarities.append(sim)

    # 在相似度低于阈值处切分
    chunks = []
    current_chunk = [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len("".join(current_chunk)) >= min_chunk_size:
            chunks.append("".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


# 使用示例
model = SentenceTransformer("all-MiniLM-L6-v2")
text = (
    "RAG 系统通过检索外部知识来增强大模型的生成能力。"
    "它能有效减少模型幻觉，提升答案的准确性。"
    "向量数据库是 RAG 系统的核心组件之一。"
    "FAISS 和 Milvus 是两种常用的向量数据库。"
    "今天的天气非常好，适合户外运动。"
    "跑步和游泳都是很好的有氧运动方式。"
    "回到技术话题，Embedding 模型的选择对 RAG 效果影响很大。"
)

chunks = semantic_chunking(text, model, threshold=0.45)
print(f"语义分块结果 ({len(chunks)} 块):")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk}")
```

---

## 4. 工程师视角

### 4.1 Embedding 模型选型指南

| 模型 | 维度 | 语言 | 许可 | 适用场景 |
|------|------|------|------|---------|
| all-MiniLM-L6-v2 | 384 | 英文 | Apache 2.0 | 原型开发、低资源环境 |
| BGE-large-zh-v1.5 | 1024 | 中文 | MIT | 中文语义检索 |
| BGE-M3 | 1024 | 100+ 语言 | MIT | 多语言检索，支持稀疏+稠密 |
| GTE-Qwen2-7B | 变长 | 多语言 | Apache 2.0 | 高质量检索，需 GPU |
| text-embedding-3-small | 1536 | 多语言 | 商业 | 快速集成，OpenAI 生态 |
| text-embedding-3-large | 3072 | 多语言 | 商业 | 最高质量，成本较高 |
| Cohere embed-v3 | 1024 | 多语言 | 商业 | 支持 search/classify 不同模式 |

**选型建议**：
- 开源优先 + 中文场景 → BGE 系列
- 多语言 + 开源 → BGE-M3 或 GTE-Qwen2
- 快速集成 + 不差钱 → OpenAI text-embedding-3
- 极致性能 + 有 GPU → GTE-Qwen2-7B

### 4.2 向量数据库选型决策树

```
你的向量规模是多少？
│
├── < 10 万：Chroma / pgvector（轻量即可）
│
├── 10 万 ~ 1000 万：
│   ├── 想要零运维 → Pinecone / Zilliz Cloud
│   ├── 已有 PostgreSQL → pgvector
│   └── 本地部署 → FAISS / Weaviate
│
└── > 1000 万：
    ├── 分布式需求 → Milvus（自托管）/ Zilliz Cloud
    └── 嵌入式 + 极致性能 → FAISS + 自建服务
```

### 4.3 FAISS 索引选型速查表

| 索引 | 适用场景 | 内存 | 速度 | 精度 |
|------|---------|------|------|------|
| IndexFlatIP/L2 | < 100K 向量 | 高 | 慢 | 100% |
| IndexIVFFlat | 100K~10M | 高 | 快 | 95%+ |
| IndexHNSWFlat | 100K~10M | 最高 | 最快 | 98%+ |
| IndexIVFPQ | > 10M（内存受限） | 低 | 快 | 90%+ |
| IndexIVFScalarQuantizer | > 10M（精度优先） | 中 | 快 | 95%+ |

**常用组合**：
```python
# 中小规模（< 1M），追求精度
index = faiss.IndexFlatIP(dim)

# 大规模（1M~100M），平衡速度和精度
index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist=1024)

# 超大规模（> 100M），内存受限
index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist=4096, m=64, nbits=8)

# 追求最快查询速度
index = faiss.IndexHNSWFlat(dim, M=32)
```

### 4.4 Chunking 的工程实践

**Parent-Child 分块模式**（推荐）：

```
大块（Parent）：用于送入 LLM（上下文完整）
  └── 小块（Child）：用于向量检索（精度高）

检索流程：
  查询 → 向量检索 → 命中 Child 块 → 找到对应 Parent 块 → Parent 块送入 LLM
```

**Chunk 元数据的重要性**：

每个 Chunk 应保存以下元数据，以支持过滤、溯源和权限控制：

```python
{
    "chunk_id": "abc123",
    "content": "...",
    "source_file": "product_manual_v2.pdf",
    "page_number": 15,
    "section_title": "安装指南",
    "created_at": "2025-01-15",
    "access_level": "internal",
    "parent_chunk_id": "xyz789",  # Parent-Child 模式
}
```

### 4.5 Embedding 推理优化

| 优化手段 | 效果 | 适用场景 |
|---------|------|---------|
| **批量编码** | 3~10x 加速 | 索引构建（离线） |
| **GPU 推理** | 10~50x 加速 | 有 GPU 的环境 |
| **ONNX Runtime** | 2~4x 加速 | CPU 推理优化 |
| **量化（INT8）** | 内存减半，速度提升 | 边缘部署 |
| **缓存** | 避免重复计算 | 高频重复查询 |
| **降维（Matryoshka）** | 减少存储和计算 | 大规模索引 |

### 4.6 常见面试 / 设计问题

| 问题 | 核心要点 |
|------|---------|
| 余弦相似度和内积的区别？ | 归一化后等价；未归一化时内积受向量长度影响 |
| HNSW 和 IVF 怎么选？ | HNSW 更快但内存大；IVF 内存更友好但需要训练 |
| Chunk 太大或太小会怎样？ | 太大引入噪声稀释相关性；太小丢失上下文 |
| 如何处理表格/图片？ | 表格转 Markdown/描述文字；图片用 VLM 生成文字描述后索引 |
| 多语言检索怎么做？ | 用多语言 Embedding 模型（BGE-M3）统一向量空间 |
| 增量更新怎么实现？ | FAISS: 重建索引；Milvus/Chroma: 原生支持增删改查 |

---

## 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| **句子 Embedding** | 通过对比学习训练，将整段文本编码为一个稠密向量 |
| **余弦相似度** | 归一化后的内积，衡量两个向量方向的一致性 |
| **ANN 搜索** | 牺牲少量精度换取数量级的速度提升（IVF / HNSW / PQ） |
| **Chunking** | 分块粒度决定检索精度，递归字符分块是稳健的默认选择 |
| **Parent-Child** | 小块检索 + 大块上下文，兼顾精度和完整性 |
| **向量数据库选型** | 小规模用 Chroma，中规模用 FAISS，大规模用 Milvus |

**下一节**：高级 RAG 技术——当 Naive RAG 效果不够时，如何通过 Query 改写、Reranker、混合检索持续提升？
