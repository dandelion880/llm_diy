import regex as re
from collections import Counter,defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

def calculate_frequency(t_list,weights):
    frequency = Counter()
    for tokens,weight in zip(t_list,weights):
        for pair in zip(tokens,tokens[1:]):
            frequency[pair]+=weight
    return frequency

def combine(tokens,best_pair):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i+1<len(tokens) and (tokens[i],tokens[i+1]) == best_pair:
            new_tokens.append(tokens[i]+tokens[i+1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def train_bpe(input_path,vocab_size,special_tokens):
    vocab = {} #用字典创建初始词表
    for i in range (256):
        vocab[i] = bytes([i])

    # 添加 special tokens
    for special_token in special_tokens:
        special_bytes = special_token.encode("utf-8")
        if special_bytes not in vocab.values():
            new_id = len(vocab)
            vocab[new_id] = special_bytes

    # 定义预分词规则
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 初始化频率统计
    t_number = Counter()

    # 构建 special token 分割规则
    if special_tokens:
        pattern = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
    else:
        pattern = None

    # 分块读取文件
    with open(input_path, "rb") as f:
        num_chunks = 4

        boundaries = find_chunk_boundaries(
            f, num_chunks, b"<|endoftext|>"
        )

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # 按 special token 分割当前 chunk
            if pattern:
                texts = re.split(pattern, chunk)
            else:
                texts = [chunk]
            # 两次按specialtoken分割的目的不同，第一次是分块读取，节省内存等
            # 第二次是保护特殊token，为它独立分配id

            # 逐个统计 pre-token
            for text in texts:
                t_number.update(re.findall(PAT, text))


    t_list = [] #每次出现的 pre-token 对应的字节 token 序列

    for t in t_number:
        raw_bytes = t.encode("utf-8")
        tokens = list(map(lambda x: bytes([x]), raw_bytes))
        t_list.append(tokens)
    weights = list(t_number.values())

    pair_frequency = Counter()
    pair_to_indices = defaultdict(set)

    for i,tokens in enumerate(t_list):
        pair_counts = Counter(zip(tokens,tokens[1:]))
        for pair,count in pair_counts.items():
            pair_frequency[pair]+=count*weights[i]
            pair_to_indices[pair].add(i)

    merges = []

    while len(vocab)<vocab_size: #training_loop
        if not pair_frequency:
            break
        best_pair = max(pair_frequency,key=lambda pair:(pair_frequency[pair],pair))
        affected_indices = list(pair_to_indices[best_pair])

        for i in affected_indices:
            tokens = t_list[i]
            weight = weights[i]

            old_counts = Counter(zip(tokens,tokens[1:])) #移除旧pair的统计
            for pair,count in old_counts.items():
                pair_frequency[pair] -= count*weight
                pair_to_indices[pair].discard(i)

                if pair_frequency[pair] == 0:
                    del pair_frequency[pair]

            new_tokens = combine(tokens,best_pair) #执行合并
            t_list[i] = new_tokens

            new_counts = Counter(zip(new_tokens,new_tokens[1:])) #添加新pair的统计

            for pair,count in new_counts.items():
                pair_frequency[pair] += count*weight
                pair_to_indices[pair].add(i)

        new_id = len(vocab)
        vocab[new_id] = best_pair[0]+best_pair[1]
        merges.append(best_pair)

    return vocab, merges