import pickle
import regex as re

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.merges = merges
        self.special_tokens = [] if special_tokens is None else special_tokens
        self.special_tokens.sort(key=len, reverse=True)
        self.byte2int = {}
        self.int2byte = {}
        for i,j in vocab.items():
            self.byte2int[j] = i
            self.int2byte[i] = j

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pickle.load(vocab_filepath)
        merges = pickle.load(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        lst = [text]

        # Ugly implementation
        # But generalizes to any weird special token
        for token in self.special_tokens:
            new_lst = []
            for chunk in lst:
                if isinstance(chunk, int):
                    new_lst.append(chunk)
                    continue

                # Forgive me for writing this dumpster fire
                subchunks = chunk.split(token)
                for subchunk in subchunks[:-1]:
                    if subchunk != "":
                        new_lst.append(subchunk)
                    new_lst.append(self.byte2int[token.encode()])
                if subchunks[-1] != "":
                    new_lst.append(subchunks[-1])
            lst = new_lst
        
        out = []
        GPT2_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in lst:
            if isinstance(chunk, int):
                out.append(chunk)
                continue
            tokens = re.finditer(GPT2_REGEX, chunk)
            parsed = []
            for token in tokens:
                parsed.append([i.to_bytes(1, 'big') for i in token.group().encode('utf-8')])
            
            for a,b in self.merges:
                new_parsed = []
                for sub_parsed in parsed:
                    new_sub_parsed = []
                    i = 0
                    while i<len(sub_parsed):

                        if i < len(sub_parsed)-1 and sub_parsed[i] == a and sub_parsed[i+1] == b:
                            new_sub_parsed.append(sub_parsed[i]+sub_parsed[i+1])
                            i += 2
                        
                        else:
                            new_sub_parsed.append(sub_parsed[i])
                            i += 1

                    new_parsed.append(new_sub_parsed)
                parsed = new_parsed
            
            for sub_parsed in parsed:
                for token in sub_parsed:
                    out.append(self.byte2int[token])
        return out
    
    def decode(self, tokens):
        ret = b''
        for t in tokens:
            ret += self.int2byte[t]
        return ret.decode('utf-8', errors='replace')



            


        



            

