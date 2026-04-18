# 数据工程与 Tokenizer

> **前置知识**：[混合精度与框架](2_混合精度与框架.md)

---

## 直觉与概述

### 数据是大模型的"燃料"

一个直觉：GPT-3 的架构跟 GPT-2 几乎一样，但 GPT-3 用了 300B tokens（约 570GB 文本），而 GPT-2 只用了 40GB。Llama-2 的架构也没什么独创，但用了 2T tokens 训练。**数据的规模和质量直接决定了模型的能力。**

关键数字对比：

```
模型           参数量      训练数据量       数据/参数比
GPT-2          1.5B       ~10B tokens     ~6.7x
GPT-3          175B       300B tokens     ~1.7x    ← 数据不足！
Chinchilla     70B        1.4T tokens     20x      ← 数据充足
Llama-1        65B        1.4T tokens     ~21x
Llama-2        70B        2T tokens       ~29x
Llama-3        405B       15T tokens      ~37x     ← 数据极其充足
```

Chinchilla 论文（2022）指出：GPT-3 是"参数过多、数据不足"——如果用同样的算力，训练一个 **更小但吃更多数据** 的模型，效果更好。这彻底改变了大模型训练的数据策略。

### Tokenizer：模型看到的不是文字

模型不直接处理文字，而是处理 **token**——一种介于字符和单词之间的子词单元：

```
原文: "Transformer模型的训练效率很高"

字符级:   T r a n s f o r m e r 模 型 的 训 练 效 率 很 高      (20 tokens)
BPE:      Trans former 模型 的 训练 效率 很高                    (8 tokens)
词级:     Transformer 模型 的 训练效率 很高                       (5 tokens，但词表太大)
```

BPE（Byte Pair Encoding）是目前最主流的分词方法：从字符开始，反复合并最频繁的相邻对，直到词表达到目标大小。

---

## 严谨定义与原理

### 1. 预训练数据收集

**典型数据来源**：

| 数据源 | 规模 | 质量 | 代表 |
|--------|------|------|------|
| 网页爬取 | 数十TB | 低-中 | Common Crawl |
| 书籍 | 几百GB | 高 | Books3, Gutenberg |
| 代码 | 几百GB | 高 | GitHub, The Stack |
| 百科 | 几十GB | 很高 | Wikipedia |
| 学术论文 | 几十GB | 很高 | ArXiv, S2ORC |
| 对话/论坛 | 几百GB | 中 | Reddit, StackOverflow |

**Common Crawl 处理流水线**（以 Llama 系列为参考）：

```
Common Crawl 原始数据 (~300TB/月)
    │
    ├─ 1. 语言识别 (fastText) → 保留目标语言
    │
    ├─ 2. URL/域名过滤 → 去除色情、暴力、钓鱼网站
    │
    ├─ 3. 规则清洗
    │    ├─ 去除 HTML 标签
    │    ├─ 去除导航栏/页脚/广告
    │    ├─ 去除短文本 (<100 字)
    │    ├─ 去除高重复率文本 (重复行 > 30%)
    │    └─ Unicode 规范化
    │
    ├─ 4. 质量过滤
    │    ├─ 困惑度过滤 (用小 LM 打分，去除乱码)
    │    ├─ 规则打分 (标点比例、大写比例、数字比例等)
    │    └─ 分类器打分 (用 Wikipedia 做正例训练二分类器)
    │
    ├─ 5. 去重
    │    ├─ 精确去重: URL/文档哈希
    │    ├─ 模糊去重: MinHash + LSH (Locality-Sensitive Hashing)
    │    └─ 子串去重: Suffix Array
    │
    └─ 6. 个人信息过滤 (PII)
         ├─ 邮箱、电话号码
         ├─ 身份证号、银行卡号
         └─ 替换为特殊 token 或删除
```

### 2. 数据配比 (Data Mixture)

不同来源的数据以不同比例混合，这是一个核心超参数。

**典型配比**（GPT-3 风格）：

| 数据源 | 占比 | 采样权重 | 训练 epochs |
|--------|------|---------|------------|
| Common Crawl (清洗后) | 60% | 0.44 | 0.44 |
| WebText2 | 22% | 2.9 | 2.9 |
| Books1 + Books2 | 8% | 1.9 | 1.9 |
| Wikipedia | 3% | 3.4 | 3.4 |
| Code | 7% | — | — |

注意：**高质量数据被多次采样**（epochs > 1），低质量数据被降权采样。Wikipedia 虽然只占总数据的 3%，但训练了 3.4 个 epoch。

**Llama-3 的数据策略演进**：
- 数据量大幅增加到 15T tokens
- 代码数据比例提高（约 17%）
- 多语言数据增加
- 引入"退火"（Annealing）：训练末期切换到纯高质量数据

**数据配比的影响**：

```
代码数据多 → 模型推理能力强、格式化输出好
书籍数据多 → 长文本理解好、文学能力强
网页数据多 → 知识覆盖广、但可能质量不均
对话数据多 → 对话能力好、但可能降低严谨性
```

### 3. Tokenizer: BPE 算法

**BPE (Byte Pair Encoding)** 的训练过程：

```
Step 0: 初始词表 = 所有单个字符 (或字节)
        语料: "low low lower newest"
        表示: l o w _ l o w _ l o w e r _ n e w e s t

Step 1: 统计所有相邻对的频率
        (l,o)=3, (o,w)=3, (w,_)=2, (w,e)=2, (e,r)=1, (e,s)=1, ...
        合并最频繁的: (l,o) → lo
        词表: {所有字符, lo}

Step 2: 重新表示: lo w _ lo w _ lo w e r _ n e w e s t
        统计: (lo,w)=3, (w,_)=2, ...
        合并: (lo,w) → low
        词表: {所有字符, lo, low}

Step 3: low _ low _ low e r _ n e w e s t
        统计: (low,_)=2, (e,w)=1, (e,r)=1, (e,s)=1, ...
        合并: (low,_) → low_
        ...

重复直到词表达到目标大小 (如 32000, 50000, 128000)
```

**Byte-level BPE**（GPT-2、GPT-4、Llama 使用）：
- 不从字符出发，而是从 **256 个字节** 出发
- 优点：任何文本都能表示（包括未知语言、emoji、二进制数据）
- 不需要 `<UNK>` token

### 4. SentencePiece

SentencePiece（Google 开发）是另一种主流 Tokenizer：

```
BPE:           先分词（按空格），再在词内做 BPE
SentencePiece: 直接在原始文本上做，空格也当成普通字符
               空格用 ▁ (U+2581) 表示
```

**优势**：
- 语言无关（不依赖空格分词，适合中日韩等语言）
- 可逆（能从 token 精确还原原文）
- 支持 BPE 和 Unigram 两种算法

**Unigram 模型**（SentencePiece 的另一种模式）：
- 从一个大词表开始，逐步删除影响最小的子词
- 基于概率模型：选择使得语料似然最高的分词方式
- 与 BPE 的"贪心合并"不同，Unigram 基于全局最优

### 5. 课程学习 (Curriculum Learning)

**核心思想**：训练顺序也很重要——先学简单的，再学难的。

```
传统方式: 所有数据随机打乱，从头训到尾
课程学习:
  Phase 1 (0-30%):   短文本、简单语法、高质量数据
  Phase 2 (30-70%):  中等长度、混合质量、增加代码
  Phase 3 (70-90%):  长文本、复杂推理、全部数据
  Phase 4 (90-100%): 退火阶段，纯高质量数据，降低 lr
```

**实践中的应用**：
- **序列长度递增**：先用短序列训练（省算力），后期增加到完整长度
- **数据质量递增**：训练末期切换到纯高质量数据（"退火"）
- **领域递增**：先通用知识，后加入领域数据

---

## Python 代码示例

### 示例 1：从零实现 BPE Tokenizer

```python
"""
从零实现 Byte Pair Encoding (BPE) Tokenizer
包括训练和编解码
"""
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
import re


class SimpleBPE:
    """简化版 BPE Tokenizer"""

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []       # 合并规则列表
        self.vocab: Dict[int, str] = {}                # id → token
        self.token_to_id: Dict[str, int] = {}          # token → id

    def _get_pairs(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """统计所有相邻 token 对的频率"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """应用一次合并操作"""
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, text: str):
        """
        训练 BPE
        Step 1: 统计词频（按空格简单分词）
        Step 2: 将每个词拆为字符序列
        Step 3: 反复合并最频繁的相邻对
        """
        # 统计词频
        words = text.split()
        word_counts = Counter(words)

        # 每个词拆为字符序列，末尾加 </w> 标记词边界
        word_freqs: Dict[Tuple[str, ...], int] = {}
        for word, count in word_counts.items():
            chars = tuple(list(word) + ["</w>"])
            word_freqs[chars] = count

        # 初始词表 = 所有单个字符
        all_chars = set()
        for word in word_freqs:
            for ch in word:
                all_chars.add(ch)

        print(f"初始词表大小: {len(all_chars)}")
        print(f"目标词表大小: {self.vocab_size}")
        print(f"需要执行 {self.vocab_size - len(all_chars)} 次合并\n")

        # 反复合并
        num_merges = self.vocab_size - len(all_chars)
        for i in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            if i < 10 or (i + 1) % 50 == 0:
                print(f"  合并 {i+1}: '{best_pair[0]}' + '{best_pair[1]}' → "
                      f"'{best_pair[0] + best_pair[1]}' (频率: {pairs[best_pair]})")

        # 构建最终词表
        vocab_tokens = sorted(all_chars)
        for pair in self.merges:
            vocab_tokens.append(pair[0] + pair[1])

        self.vocab = {i: token for i, token in enumerate(vocab_tokens)}
        self.token_to_id = {token: i for i, token in self.vocab.items()}
        print(f"\n最终词表大小: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 序列"""
        tokens = []
        for word in text.split():
            # 拆为字符
            word_tokens = list(word) + ["</w>"]
            # 按训练时的顺序应用合并规则
            for merge_pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                        word_tokens[i] = merge_pair[0] + merge_pair[1]
                        del word_tokens[i + 1]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return [self.token_to_id.get(t, 0) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """将 token id 序列解码回文本"""
        tokens = [self.vocab.get(i, "?") for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()


def main():
    # 训练语料
    corpus = """
    the cat sat on the mat the cat ate the rat the dog sat on the log
    the cat and the dog are friends the cat likes fish the dog likes bones
    the neural network learns patterns the transformer model uses attention
    the large language model generates text the training data is important
    """ * 10  # 重复增加频率

    # 训练 BPE
    bpe = SimpleBPE(vocab_size=80)
    bpe.train(corpus)

    # 编码/解码测试
    test_text = "the cat sat on the mat"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)

    print(f"\n原文: '{test_text}'")
    print(f"编码: {encoded}")
    print(f"Token: {[bpe.vocab[i] for i in encoded]}")
    print(f"解码: '{decoded}'")

    # 压缩率
    char_count = len(test_text.replace(" ", "")) + test_text.count(" ")  # 带空格
    token_count = len(encoded)
    print(f"\n压缩率: {char_count} 字符 → {token_count} tokens "
          f"(平均 {char_count/token_count:.1f} 字符/token)")


if __name__ == "__main__":
    main()
```

### 示例 2：使用 HuggingFace Tokenizers 训练 BPE

```python
"""
使用 HuggingFace tokenizers 库训练 BPE Tokenizer
这是实际工程中的标准做法
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFKC, Sequence as NormSeq


def train_bpe_tokenizer(
    files: list = None,
    vocab_size: int = 32000,
    min_frequency: int = 2,
):
    """
    训练 Byte-level BPE Tokenizer (类似 GPT-2/Llama)

    参数:
        files: 训练文件路径列表
        vocab_size: 目标词表大小
        min_frequency: 最小合并频率
    """
    # 1. 选择底层模型: BPE
    tokenizer = Tokenizer(models.BPE())

    # 2. 预处理: Byte-level 预分词
    # 将文本按空格和标点初步分割，然后转为字节表示
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 3. 规范化
    tokenizer.normalizer = NormSeq([NFKC()])  # Unicode 规范化

    # 4. 解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 5. 训练器配置
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<|begin_of_text|>",    # BOS
            "<|end_of_text|>",      # EOS
            "<|pad|>",              # PAD
            "<|unk|>",              # UNK (byte-level BPE 通常不需要)
            "<|start_header_id|>",  # Chat 模板
            "<|end_header_id|>",    # Chat 模板
        ],
        show_progress=True,
    )

    # 6. 训练
    if files:
        tokenizer.train(files, trainer)
    else:
        # 用示例文本演示
        sample_texts = [
            "The Transformer architecture has revolutionized natural language processing.",
            "Large language models like GPT-4 and Llama are trained on trillions of tokens.",
            "混合精度训练使用 FP16 或 BF16 减少显存占用。",
            "分布式训练需要数据并行和模型并行的配合。",
            "def train_model(model, data, epochs=3):\n    for epoch in range(epochs):\n        loss = model(data)\n",
        ] * 100  # 重复以获得足够的统计量

        tokenizer.train_from_iterator(sample_texts, trainer)

    return tokenizer


def analyze_tokenizer(tokenizer):
    """分析 Tokenizer 的表现"""
    test_cases = [
        "Hello, world!",
        "The Transformer model uses self-attention.",
        "大模型训练需要分布式计算。",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "GPT-4 has 1.8T parameters (rumored).",
        "https://arxiv.org/abs/2005.14165",
    ]

    print(f"\n词表大小: {tokenizer.get_vocab_size()}")
    print(f"{'='*70}")

    for text in test_cases:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        ids = encoded.ids

        print(f"\n原文: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")

        # 解码验证
        decoded = tokenizer.decode(ids)
        match = "OK" if decoded == text else f"MISMATCH: '{decoded}'"
        print(f"解码验证: {match}")

        # 压缩率
        chars_per_token = len(text) / len(tokens)
        print(f"压缩率: {chars_per_token:.2f} 字符/token")


def compare_tokenizers():
    """对比不同模型的 Tokenizer"""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("需要安装 transformers: pip install transformers")
        return

    models_to_compare = [
        ("gpt2", "GPT-2 (50257)"),
        ("meta-llama/Llama-2-7b-hf", "Llama-2 (32000)"),
        # ("meta-llama/Meta-Llama-3-8B", "Llama-3 (128000)"),
    ]

    test_text = "The quick brown fox jumps over the lazy dog. 大模型训练非常消耗算力。"

    print(f"\n测试文本: '{test_text}'")
    print(f"{'='*70}")

    for model_name, display_name in models_to_compare:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(test_text)
            ids = tokenizer.encode(test_text)
            print(f"\n{display_name}:")
            print(f"  Token 数: {len(tokens)}")
            print(f"  Tokens: {tokens[:15]}...")
            print(f"  压缩率: {len(test_text)/len(tokens):.2f} 字符/token")
        except Exception as e:
            print(f"\n{display_name}: 加载失败 ({e})")


if __name__ == "__main__":
    tokenizer = train_bpe_tokenizer(vocab_size=1000)
    analyze_tokenizer(tokenizer)
    # compare_tokenizers()  # 需要 transformers 和模型权限
```

### 示例 3：数据清洗与去重流水线

```python
"""
预训练数据清洗流水线示例
包含: 规则过滤、质量评分、MinHash 去重
"""
import re
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


class TextCleaner:
    """基于规则的文本清洗器"""

    def __init__(self):
        self.filters = [
            self._filter_too_short,
            self._filter_too_many_special_chars,
            self._filter_high_repetition,
            self._filter_low_alpha_ratio,
            self._remove_html_tags,
            self._normalize_whitespace,
        ]

    def clean(self, text: str) -> Optional[str]:
        """
        应用所有清洗规则
        返回 None 表示应丢弃该文档
        """
        for filter_fn in self.filters:
            result = filter_fn(text)
            if result is None:
                return None
            text = result
        return text

    def _filter_too_short(self, text: str) -> Optional[str]:
        """丢弃过短的文档"""
        if len(text.split()) < 50:  # 少于 50 词
            return None
        return text

    def _filter_too_many_special_chars(self, text: str) -> Optional[str]:
        """丢弃特殊字符过多的文档（可能是乱码）"""
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.3:
            return None
        return text

    def _filter_high_repetition(self, text: str) -> Optional[str]:
        """丢弃高重复率文档"""
        lines = text.split("\n")
        if len(lines) < 3:
            return text

        # 检查重复行比例
        unique_lines = set(line.strip() for line in lines if line.strip())
        if len(unique_lines) < len(lines) * 0.5:  # 超过 50% 重复
            return None

        # 检查重复 n-gram
        words = text.lower().split()
        if len(words) < 20:
            return text

        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        most_common_freq = bigram_counts.most_common(1)[0][1]
        if most_common_freq / len(bigrams) > 0.1:  # 最常见二元组占比 > 10%
            return None

        return text

    def _filter_low_alpha_ratio(self, text: str) -> Optional[str]:
        """丢弃字母比例过低的文档"""
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / max(len(text), 1) < 0.4:
            return None
        return text

    def _remove_html_tags(self, text: str) -> str:
        """移除残留的 HTML 标签"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # HTML 实体
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class MinHashDeduplicator:
    """
    MinHash + LSH 模糊去重
    用于检测近似重复的文档
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.8,
                 ngram_size: int = 5):
        """
        参数:
            num_perm: MinHash 签名的维度数
            threshold: Jaccard 相似度阈值，超过则认为重复
            ngram_size: n-gram 大小
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram_size = ngram_size
        # 简化: 用多组哈希种子模拟多个哈希函数
        self.hash_seeds = list(range(num_perm))
        self.signatures: Dict[str, List[int]] = {}  # doc_id → MinHash 签名

    def _get_ngrams(self, text: str) -> set:
        """提取字符 n-gram"""
        text = text.lower().strip()
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams

    def _compute_minhash(self, ngrams: set) -> List[int]:
        """计算 MinHash 签名"""
        signature = []
        for seed in self.hash_seeds:
            min_hash = float('inf')
            for ngram in ngrams:
                h = int(hashlib.md5(
                    f"{seed}:{ngram}".encode()
                ).hexdigest(), 16) % (2**32)
                min_hash = min(min_hash, h)
            signature.append(min_hash if min_hash != float('inf') else 0)
        return signature

    def _estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        """用 MinHash 签名估算 Jaccard 相似度"""
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    def add_document(self, doc_id: str, text: str):
        """添加文档到索引"""
        ngrams = self._get_ngrams(text)
        if ngrams:
            self.signatures[doc_id] = self._compute_minhash(ngrams)

    def is_duplicate(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        检查文档是否与已有文档重复
        返回: (是否重复, 重复文档ID)
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return False, None

        new_sig = self._compute_minhash(ngrams)

        for existing_id, existing_sig in self.signatures.items():
            similarity = self._estimate_jaccard(new_sig, existing_sig)
            if similarity >= self.threshold:
                return True, existing_id

        return False, None


class QualityScorer:
    """文档质量评分器"""

    def score(self, text: str) -> float:
        """
        对文档质量打分 (0-1)
        实际工程中会用训练好的分类器
        这里用启发式规则演示
        """
        scores = []

        # 1. 长度分数
        word_count = len(text.split())
        length_score = min(word_count / 200, 1.0)  # 200 词以上满分
        scores.append(("长度", length_score, 0.2))

        # 2. 标点规范性
        sentences = re.split(r'[.!?。！？]', text)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        punct_score = 1.0 if 5 < avg_sentence_len < 40 else 0.5
        scores.append(("标点", punct_score, 0.2))

        # 3. 词汇丰富度 (Type-Token Ratio)
        words = text.lower().split()
        ttr = len(set(words)) / max(len(words), 1)
        vocab_score = min(ttr / 0.5, 1.0)
        scores.append(("词汇丰富度", vocab_score, 0.2))

        # 4. 段落结构
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        struct_score = min(len(paragraphs) / 3, 1.0)
        scores.append(("段落结构", struct_score, 0.2))

        # 5. 非重复性
        lines = text.split("\n")
        unique_ratio = len(set(lines)) / max(len(lines), 1)
        scores.append(("非重复性", unique_ratio, 0.2))

        # 加权求和
        total = sum(score * weight for _, score, weight in scores)
        return total


def run_pipeline():
    """运行完整的数据清洗流水线"""
    # 模拟文档集合
    documents = [
        ("doc1", "The Transformer architecture was introduced in 2017 by Vaswani et al. "
                 "It uses self-attention mechanisms to process sequences in parallel. "
                 "This was a major breakthrough compared to recurrent neural networks. " * 5),

        ("doc2", "Buy cheap products! Click here! Best deals! " * 20),  # 垃圾文本

        ("doc3", "The Transformer architecture was introduced in 2017 by Vaswani and colleagues. "
                 "It uses self-attention to process sequences in parallel. "
                 "This was a major advance over recurrent networks. " * 5),  # 近似重复

        ("doc4", "x" * 500),  # 重复字符

        ("doc5", "Machine learning models learn from data to make predictions. "
                 "Deep learning uses neural networks with multiple layers. "
                 "Convolutional networks excel at image recognition tasks. "
                 "Recurrent networks process sequential data like text and speech. "
                 "Transformer models use attention for parallel processing. " * 3),

        ("doc6", ""), # 空文档

        ("doc7", "分布式训练是大模型训练的核心技术。数据并行将数据分配到多个GPU上，"
                 "每个GPU持有完整的模型副本。模型并行则将模型本身切分到多个设备上。"
                 "ZeRO优化通过切分冗余的优化器状态来降低显存占用。" * 4),
    ]

    cleaner = TextCleaner()
    deduplicator = MinHashDeduplicator(num_perm=64, threshold=0.7)
    scorer = QualityScorer()

    print("="*70)
    print(" 预训练数据清洗流水线")
    print("="*70)

    kept = 0
    for doc_id, text in documents:
        print(f"\n--- {doc_id} ---")
        print(f"原始长度: {len(text)} 字符")

        # Step 1: 规则清洗
        cleaned = cleaner.clean(text)
        if cleaned is None:
            print(f"  [规则过滤] 丢弃")
            continue
        print(f"  [规则清洗] 通过 (清洗后 {len(cleaned)} 字符)")

        # Step 2: 去重
        is_dup, dup_id = deduplicator.is_duplicate(doc_id, cleaned)
        if is_dup:
            print(f"  [去重] 丢弃 (与 {dup_id} 重复)")
            continue
        deduplicator.add_document(doc_id, cleaned)
        print(f"  [去重] 通过")

        # Step 3: 质量评分
        quality = scorer.score(cleaned)
        threshold = 0.5
        if quality < threshold:
            print(f"  [质量评分] 丢弃 (分数: {quality:.2f} < {threshold})")
            continue
        print(f"  [质量评分] 通过 (分数: {quality:.2f})")

        kept += 1
        print(f"  >>> 保留")

    print(f"\n{'='*70}")
    print(f"总计: {len(documents)} 篇 → 保留 {kept} 篇 ({kept/len(documents)*100:.0f}%)")


if __name__ == "__main__":
    run_pipeline()
```

### 示例 4：数据配比与采样器

```python
"""
多源数据配比采样器
按指定比例从不同数据源采样
"""
import random
from typing import Dict, List, Iterator
from dataclasses import dataclass


@dataclass
class DataSource:
    """数据源"""
    name: str
    documents: List[str]
    weight: float           # 采样权重
    quality: str = "medium" # low / medium / high


class WeightedDataSampler:
    """
    加权数据采样器
    按照指定权重从多个数据源采样

    支持:
    - 按权重采样（高质量数据多采几遍）
    - 数据用尽后的处理策略
    - 课程学习（训练阶段性调整权重）
    """

    def __init__(self, sources: List[DataSource], seed: int = 42):
        self.sources = sources
        self.rng = random.Random(seed)
        self._normalize_weights()

    def _normalize_weights(self):
        """归一化权重"""
        total = sum(s.weight for s in self.sources)
        for s in self.sources:
            s.weight /= total

    def sample(self, n_samples: int) -> Iterator[Dict]:
        """按权重采样 n 条数据"""
        # 计算每个源的采样数
        source_counts = {}
        for s in self.sources:
            source_counts[s.name] = int(n_samples * s.weight)

        # 处理余数
        remaining = n_samples - sum(source_counts.values())
        for s in self.sources[:remaining]:
            source_counts[s.name] += 1

        # 从每个源采样
        all_samples = []
        for source in self.sources:
            count = source_counts[source.name]
            n_docs = len(source.documents)
            epochs = count / n_docs if n_docs > 0 else 0

            for i in range(count):
                doc_idx = i % n_docs  # 数据不够就循环（多 epoch）
                all_samples.append({
                    "text": source.documents[doc_idx],
                    "source": source.name,
                    "epoch": i // n_docs,
                })

        # 打乱
        self.rng.shuffle(all_samples)
        yield from all_samples

    def curriculum_schedule(
        self,
        total_steps: int,
        phases: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """
        课程学习调度: 不同训练阶段使用不同的数据配比

        参数:
            total_steps: 总训练步数
            phases: 每个阶段的权重配置
                    [{"web": 0.8, "wiki": 0.2}, {"web": 0.5, "wiki": 0.3, "code": 0.2}]

        返回: 每一步的权重
        """
        steps_per_phase = total_steps // len(phases)
        schedule = []

        for phase_idx, phase_weights in enumerate(phases):
            for step in range(steps_per_phase):
                # 线性插值到下一阶段
                if phase_idx < len(phases) - 1:
                    progress = step / steps_per_phase
                    next_weights = phases[phase_idx + 1]
                    blended = {}
                    for key in set(list(phase_weights.keys()) + list(next_weights.keys())):
                        w1 = phase_weights.get(key, 0)
                        w2 = next_weights.get(key, 0)
                        blended[key] = w1 * (1 - progress) + w2 * progress
                    schedule.append(blended)
                else:
                    schedule.append(phase_weights)

        return schedule


def main():
    # 定义数据源
    sources = [
        DataSource("web_crawl", [f"Web document {i}" for i in range(1000)],
                    weight=0.60, quality="medium"),
        DataSource("wikipedia", [f"Wiki article {i}" for i in range(100)],
                    weight=0.15, quality="high"),
        DataSource("books", [f"Book chapter {i}" for i in range(200)],
                    weight=0.10, quality="high"),
        DataSource("code", [f"Code file {i}" for i in range(300)],
                    weight=0.10, quality="high"),
        DataSource("arxiv", [f"Paper {i}" for i in range(50)],
                    weight=0.05, quality="high"),
    ]

    sampler = WeightedDataSampler(sources)

    # 采样统计
    n_samples = 10000
    source_counts = {}
    epoch_counts = {}

    for sample in sampler.sample(n_samples):
        src = sample["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        epoch = sample["epoch"]
        if src not in epoch_counts:
            epoch_counts[src] = []
        epoch_counts[src].append(epoch)

    print(f"总采样: {n_samples} 条")
    print(f"\n{'数据源':<15} {'采样数':>8} {'占比':>8} {'文档数':>8} {'训练轮次':>10}")
    print("-" * 55)

    for src in sources:
        count = source_counts.get(src.name, 0)
        max_epoch = max(epoch_counts.get(src.name, [0])) + 1
        print(f"{src.name:<15} {count:>8} {count/n_samples:>7.1%} "
              f"{len(src.documents):>8} {max_epoch:>10.1f}")

    # 课程学习调度
    print(f"\n{'='*60}")
    print("课程学习调度示例:")
    phases = [
        {"web_crawl": 0.8, "wikipedia": 0.1, "books": 0.05, "code": 0.05},
        {"web_crawl": 0.5, "wikipedia": 0.15, "books": 0.1, "code": 0.15, "arxiv": 0.1},
        {"web_crawl": 0.3, "wikipedia": 0.2, "books": 0.15, "code": 0.2, "arxiv": 0.15},
    ]

    schedule = sampler.curriculum_schedule(total_steps=30, phases=phases)

    print(f"\n{'步骤':>4} | {'web':>6} | {'wiki':>6} | {'books':>6} | {'code':>6} | {'arxiv':>6}")
    print("-" * 50)
    for i in [0, 5, 9, 10, 15, 19, 20, 25, 29]:
        if i < len(schedule):
            w = schedule[i]
            print(f"{i:>4} | {w.get('web_crawl', 0):>5.1%} | {w.get('wikipedia', 0):>5.1%} | "
                  f"{w.get('books', 0):>5.1%} | {w.get('code', 0):>5.1%} | {w.get('arxiv', 0):>5.1%}")


if __name__ == "__main__":
    main()
```

---

## 工程视角

### Tokenizer 设计的工程决策

```
词表大小选择:
  太小 (8K):   中文每个字拆成多个 byte token → 效率极低
  太大 (256K): Embedding 层参数暴增 → 显存浪费
  常见选择:
    GPT-2:     50257
    GPT-4:     100277 (cl100k_base)
    Llama-2:   32000
    Llama-3:   128256  ← 大词表，提升多语言和代码效率

词表大小的影响:
  - 词表越大 → 同样文本的 token 数越少 → 训练和推理越快
  - 但 Embedding 层越大 → 参数越多
  - 对于 7B+ 模型，Embedding 占比小，大词表更优
```

### 数据去重的工程实现

```
精确去重 (URL/Hash):
  - 速度: 极快 (O(n))
  - 效果: 只能去完全相同的
  - 实现: 对文档计算 SHA-256/MD5，去掉哈希相同的

模糊去重 (MinHash + LSH):
  - 速度: 快 (O(n) 构建, O(1) 查询)
  - 效果: 能检测 ~80% 重复的文档
  - 实际参数: 128 个哈希函数, Jaccard 阈值 0.8
  - 工具: datasketch 库、Spark 实现

子串去重 (Suffix Array):
  - 速度: 慢 (O(n log n))
  - 效果: 能检测文档间共享的长段落
  - 场景: 最终精细去重

Llama-3 的去重策略:
  1. URL 精确去重
  2. MinHash (Jaccard > 0.7) 全局去重
  3. 同域名内做子串去重
  结果: Common Crawl 从 ~200TB 降到 ~15TB
```

### 数据质量评估的关键指标

```
1. 困惑度 (Perplexity):
   用小 LM (如 KenLM 5-gram) 对每篇文档打分
   困惑度太高 → 乱码/非自然语言 → 丢弃
   困惑度太低 → 可能是重复/模板文本 → 降权

2. 分类器打分:
   正例: Wikipedia 文章 (高质量)
   负例: 随机 Common Crawl 页面
   训练 fastText 分类器，对所有文档打分

3. 基于规则的指标:
   - 字母比例 > 80%
   - 平均句长 10~50 词
   - 重复行比例 < 30%
   - 有效标点比例
   - 首字母大写句子的比例

4. 人工抽查:
   随机抽样 → 人工标注质量 → 校准自动化指标
```

### 关于中文 Tokenizer 的特殊处理

```
挑战: 中文没有空格分词
解决方案:
  1. Byte-level BPE: 每个中文字符是 3 个 UTF-8 字节
     "你好" → [ä½, ł, å¥½]  (效率低，3 tokens/字)

  2. Character-aware BPE: 每个中文字符作为初始 token
     "你好" → [你, 好]  (效率高，但词表大)

  3. SentencePiece: 直接在原始文本上训练
     "你好世界" → [▁你好, 世界]  (自动学到常见词组)

Llama-3 的做法:
  - 128K 词表，包含大量中文 token
  - 中文效率比 Llama-2 (32K 词表) 提升约 2-3x
```

---

## 小结

| 主题 | 核心要点 | 工程建议 |
|------|---------|---------|
| 数据收集 | Common Crawl + 高质量源 | 多源混合，高质量数据多次采样 |
| 数据清洗 | 规则过滤 + 质量评分 + 去重 | MinHash 去重 + 困惑度过滤 |
| 数据配比 | Chinchilla 法则: 数据量 ≈ 20× 参数量 | 代码/书籍/Wiki 适当提权 |
| BPE Tokenizer | 从字节出发，贪心合并 | 词表 32K~128K，大模型用大词表 |
| SentencePiece | 语言无关，直接在原始文本上训练 | 中日韩等无空格语言首选 |
| 课程学习 | 先简后难，末期退火 | 末期切换纯高质量数据 |

> **下一节**：[对齐训练实践](4_对齐训练实践.md) — 预训练完成后，如何通过 SFT/RLHF/DPO 让模型"听话"。
