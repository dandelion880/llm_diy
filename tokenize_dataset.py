from pathlib import Path
import numpy as np

from cs336_basics.tokenizer import Tokenizer

tokenizer = Tokenizer.from_files(
    "outputs/tinystories_10k/vocab.pkl",
    "outputs/tinystories_10k/merges.pkl",
    ["<|endoftext|>"],
)

output_dir = Path("outputs/tinystories_10k")

def iter_documents(path):
    buffer = ""
    with open(path,"r",encoding="utf-8",newline="") as f:
        for line in f:
            buffer += line
            parts = buffer.split("<|endoftext|>")

            for part in parts[:-1]:
                yield part+"<|endoftext|>"

            buffer = parts[-1]
        if buffer:
            yield buffer

def tokenize_file(input_path,output_path):
    total_bytes = input_path.stat().st_size
    processed_bytes = 0
    total_tokens = 0
    next_report = 0.1
    print(f"开始编码：{input_path.name}",flush=True)

    with open(output_path,"xb") as f:
        for document in iter_documents(input_path):
            ids = tokenizer.encode(document)

            np.asarray(ids,dtype=np.uint16).tofile(f)

            total_tokens += len(ids)
            processed_bytes += len(document.encode("utf-8"))

            if processed_bytes == total_bytes*next_report:
                print(f"进度: {processed_bytes / total_bytes:.0%}", flush=True)
                next_report += 0.1

    assert processed_bytes == total_bytes
    assert output_path.stat().st_size == total_tokens*2
    print(f"完成，Token数量：{total_tokens}")

for split in ["valid", "train"]:
    input_path = Path(f"data/TinyStoriesV2-GPT4-{split}.txt")
    output_path = output_dir / f"{split}.bin"

    tokenize_file(input_path, output_path)

    