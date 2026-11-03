import time
import resource
from cs336_basics.bpe import train_bpe

import pickle
from pathlib import Path

input_path = "data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
print("开始BPE训练",flush=True)

start = time.perf_counter()

vocab,merges = train_bpe(
    input_path = input_path,
    vocab_size = vocab_size,
    special_tokens = ["<|endoftext|>"]
)

end = time.perf_counter()

peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

longest_token = max(
    (token for token in vocab.values() if token != b"<|endoftext|>"),
    key=len
)

# 创建保存目录
save_dir = Path("outputs/tinystories_10k")
save_dir.mkdir(parents=True,exist_ok=True)

# 分别在保存目录下保存vocab和merges
with open(save_dir/"vocab.pkl","wb") as f:
    pickle.dump(vocab,f)

with open(save_dir/"merges.pkl","wb") as f:
    pickle.dump(merges,f)

print(f"词表大小: {len(vocab)}")
print(f"Merge 次数: {len(merges)}")
print(f"训练时间: {end - start:.2f} 秒")
print(f"进程峰值内存: {peak_memory:.2f} MiB")
print("最长普通 token:", longest_token)
print("字节长度:", len(longest_token))
print("BPE训练结果已保存")