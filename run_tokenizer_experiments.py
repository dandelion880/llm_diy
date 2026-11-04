from pathlib import Path
import random
import time

from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files(
    "outputs/tinystories_10k/vocab.pkl",
    "outputs/tinystories_10k/merges.pkl",
    ["<|endoftext|>"]
)

valid_path = Path("data/TinyStoriesV2-GPT4-valid.txt")

# 随机采样十篇文章
text = valid_path.read_text(encoding="utf-8")
documents=[
    doc for doc in text.split("<|endoftext|>")
    if doc.strip()
]

samples = random.Random(42).sample(documents,10)

# 计算压缩率
total_bytes = sum(len(doc.encode("utf-8")) for doc in samples)
total_tokens = sum(len(tokenizer.encode(doc)) for doc in samples)

compression_ratio = total_bytes/total_tokens

print(f"压缩率：{compression_ratio:.3f} bytes/token")

# 测试编码速度
start = time.perf_counter()
with valid_path.open("r",encoding="utf-8") as f:
    token_count = sum(1 for _ in tokenizer.encode_iterable(f))

elapsed = time.perf_counter() - start
throughput = valid_path.stat().st_size/elapsed

print(f"验证集 Token 数量: {token_count}")
print(f"编码耗时: {elapsed:.2f} 秒")
print(f"编码速度: {throughput / 1e6:.3f} MB/s")

# 估算编码pile的时间
pile_days = 825e9/throughput/86400
print(f"编码 825GB Pile 预计需要: {pile_days:.2f} 天")