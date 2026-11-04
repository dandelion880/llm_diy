import regex as re
from collections.abc import Iterable,Iterator
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
            self,
            vocab:dict[int,bytes],
            merges:list[tuple[bytes,bytes]],
            special_tokens:list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        # 规定merges中记载的merge顺序，以字典形式保存
        self.merge_ranks = {
            pair:rank for rank,pair in enumerate(self.merges)
        }
        self.special_tokens = special_tokens or []

        self.bytes_to_id={
            token:idx for idx,token in self.vocab.items()
            }
        next_id = max(self.vocab,default=-1) +1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id+=1

    @classmethod
    def from_files(
        cls,
        vocab_filepath:str,
        merges_filepath:str,
        special_tokens:list[str] | None = None,
    ):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens)

    def decode(self,ids:list[int])->str:
        tokens = [self.vocab[idx] for idx in ids]
        byte_sequence = b"".join(tokens)
        return byte_sequence.decode("utf-8",errors="replace")

    def _bpe(self,piece:str)->list[int]:
        byte_tokens = [bytes([b]) for b in piece.encode("utf-8")]

        while len(byte_tokens)>1:
            candidates=[]

            for i in range(len(byte_tokens)-1):
                pair = (byte_tokens[i],byte_tokens[i+1])

                if pair in self.merge_ranks:
                    candidates.append((self.merge_ranks[pair],i))
            
            if not candidates:
                break

            _,i = min(candidates)
            byte_tokens[i:i+2] = [byte_tokens[i]+byte_tokens[i+1]]

        return [self.bytes_to_id[token] for token in byte_tokens]

    def encode(self,text:str)->list[int]:
        ids = []

        if self.special_tokens:
            special_pattern = "("+"|".join(
                re.escape(token)
                for token in sorted(self.special_tokens,key=len,reverse=True)
            )+")"
            pre_tokens = re.split(special_pattern,text)
        else:
            pre_tokens = [text]

        for pre_token in pre_tokens:
            if not pre_token:
                continue

            if pre_token in self.special_tokens:
                ids.append(self.bytes_to_id[pre_token.encode("utf-8")])
            else:
                for match in re.finditer(PAT,pre_token):
                    ids.extend(self._bpe(match.group()))

        return ids

    # 以流式方法处理文本
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for text in iterable:
            yield from self.encode(text)