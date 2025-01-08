class BEPTokenizer():

    def __init__(self,text, vocab_size:5000):
        self.max_vocab_size = vocab_size
        self.corpus = text

    def get_token_stats(self,ids):
        counts={}
        for pair in zip(ids,ids[1:]):
            counts[pair] = counts.get(pair,0)+1
        return counts
    
    def merge(self,ids,pair,idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def encode(self,text):
        tokens = list(text.encode("utf-8"))
        while len(tokens)>=2:
            stats = self.get_token_stats(tokens)
            pair = min(stats,key=lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def train_BPE_Tokenizer(self):
        self.vocab = {idx:bytes([idx]) for idx in range(256)}
        num_merges = self.max_vocab_size-256
        tokens = self.corpus.encode("utf-8") # raw bytes
        tokens = list(map(int,tokens)) #convert to a list of integers
        ids = list(tokens)
        self.merges = {}
        print(f"Before training: tokens length: {len(tokens)}")
        print("Training started...")
        for i in range(num_merges):
            stats = self.get_token_stats(ids)
            pair = max(stats,key=stats.get)
            idx = 256+i
            ids = self.merge(ids,pair,idx)
            self.merges[pair] = idx

        print("Training completed successfully!")
        for (p0,p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0]+self.vocab[p1]
        print(f"After training: tokens length: {len(ids)}")
        print(f"After training: merges length: {len(self.merges)}")
        print(f"After Training Vocab length {len(self.vocab)}")
        print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
        return self.vocab, self.merges

    def decode(self,ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf=8",errors="replace")
        return text