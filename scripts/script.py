# IntelliRAG — Komut Satırı Değerlendirme Betiği

import argparse
import json
import os
import re
import sys
import warnings
import numpy as np
from typing import List, Dict

warnings.filterwarnings('ignore')


# ── Bağımlılık kontrolü ───────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ['torch', 'sentence_transformers', 'faiss', 'rank_bm25', 'datasets']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f" Eksik kütüphaneler: {', '.join(missing)}")
        print("   Kurulum: pip install -r requirements.txt")
        sys.exit(1)

check_deps()

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss


# ── Sabitler ──────────────────────────────────────────────────────────────────
ENCODER_MODEL  = 'paraphrase-multilingual-MiniLM-L12-v2'
CE_MODEL_NAME  = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Yardımcılar ───────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Modüller ──────────────────────────────────────────────────────────────────
class SemanticChunker:
    def __init__(self, encoder, threshold=0.75, max_tok=512):
        self.enc = encoder
        self.thr = threshold
        self.max = max_tok

    def chunk(self, text: str) -> List[str]:
        sents = split_sentences(text)
        if not sents:
            return [text]
        embs  = self.enc.encode(sents, show_progress_bar=False)
        chunks, cur, cur_tok = [], [sents[0]], len(sents[0].split())
        for i in range(1, len(sents)):
            sim = cosine_sim(embs[i-1], embs[i])
            nt  = len(sents[i].split())
            if sim < self.thr or cur_tok + nt > self.max:
                chunks.append(' '.join(cur))
                cur, cur_tok = [sents[i]], nt
            else:
                cur.append(sents[i])
                cur_tok += nt
        if cur:
            chunks.append(' '.join(cur))
        return chunks


class HybridRetriever:
    def __init__(self, encoder, rrf_k=60):
        self.enc   = encoder
        self.rrf_k = rrf_k

    def build(self, chunks: List[str]):
        self.chunks = chunks
        tok         = [c.lower().split() for c in chunks]
        self.bm25   = BM25Okapi(tok)
        embs        = self.enc.encode(chunks, show_progress_bar=False, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embs)
        self.idx    = faiss.IndexFlatIP(embs.shape[1])
        self.idx.add(embs)

    def search(self, query: str, k: int = 10) -> List[str]:
        bm25_sc = self.bm25.get_scores(query.lower().split())

        q = self.enc.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')

        faiss.normalize_L2(q)

        dense_sc, _ = self.idx.search(q, len(self.chunks))
        dense_sc = dense_sc[0]

        bm25_norm = (bm25_sc - bm25_sc.min()) / (
            bm25_sc.max() - bm25_sc.min() + 1e-8
        )

        dense_norm = (dense_sc - dense_sc.min()) / (
            dense_sc.max() - dense_sc.min() + 1e-8
        )

        scores = 0.4 * bm25_norm + 0.6 * dense_norm

        top = np.argsort(scores)[::-1][:k]

        return [self.chunks[i] for i in top]
    

def rerank(query: str, contexts: List[str], ce_model, top_k: int = 5) -> List[str]:
    if not contexts:
        return contexts
    pairs  = [(query, c) for c in contexts]
    scores = ce_model.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]


def compress(query: str, contexts: List[str], encoder, ratio: float = 0.5) -> List[str]:
    sents = [s for c in contexts for s in split_sentences(c) if len(s) > 15]
    if not sents:
        return contexts
    q_emb  = encoder.encode([query], convert_to_numpy=True)[0]
    s_embs = encoder.encode(sents, convert_to_numpy=True)
    sims   = np.dot(s_embs, q_emb) / (np.linalg.norm(s_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
    n      = max(2, int(len(sents) * ratio))
    kept   = [sents[i] for i in sorted(np.argsort(sims)[::-1][:n])]
    return [' '.join(kept)]


# ── Metrikler ─────────────────────────────────────────────────────────────────
def faithfulness(answer: str, contexts: List[str]) -> float:
    words = answer.lower().split()
    ctx   = ' '.join(contexts).lower()
    return sum(1 for w in words if w in ctx) / max(len(words), 1)

def context_precision(contexts: List[str], answer: str) -> float:
    aw  = set(answer.lower().split())
    ctx = ' '.join(contexts).lower()
    return sum(1 for w in aw if w in ctx) / max(len(aw), 1)

def answer_relevancy(query: str, answer: str, encoder) -> float:
    e = encoder.encode([query, answer], convert_to_numpy=True)
    return float(np.clip(np.dot(e[0], e[1]) / (np.linalg.norm(e[0])*np.linalg.norm(e[1])+1e-8), 0, 1))


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(sample: Dict, chunker, retriever, ce_model, encoder) -> Dict:
    """Tek bir örnek üzerinde tam IntelliRAG pipeline'ını çalıştır."""
    q, ctx, ans = sample['question'], sample['context'], sample['answer']
    chunks  = chunker.chunk(ctx)
    retriever.build(chunks)
    cands   = retriever.search(q, k=10)
    ranked  = rerank(q, cands, ce_model, top_k=5)
    final   = compress(q, ranked, encoder, ratio=0.5)
    return {
        'faithfulness':      faithfulness(ans, final),
        'context_precision': context_precision(final, ans),
        'answer_relevancy':  answer_relevancy(q, ans, encoder),
        'token_count':       sum(len(c.split()) for c in final)
    }


# ── Veri yükleme ──────────────────────────────────────────────────────────────
def load_samples(n: int) -> List[Dict]:
    print(f" Natural Questions yükleniyor ({n} örnek)...")
    ds = load_dataset("natural_questions", split="validation", trust_remote_code=True)
    ds = ds.select(range(min(n * 3, len(ds))))
    samples = []
    for item in ds:
        q   = item['question']['text']
        ctx = ' '.join(item['document']['tokens']['token'][:500])
        ans = ""
        for ann in item['annotations']['short_answers']:
            if ann['text']:
                ans = ann['text'][0]
                break
        if ans:
            samples.append({'question': q, 'context': ctx, 'answer': ans})
        if len(samples) >= n:
            break
    print(f" {len(samples)} kullanılabilir örnek")
    return samples


# ── Modları ───────────────────────────────────────────────────────────────────
def mode_evaluate(args):
    """Standart değerlendirme modu."""
    samples = load_samples(args.n_samples)
    print("\n Modeller yükleniyor...")
    encoder  = SentenceTransformer(ENCODER_MODEL)
    ce_model = CrossEncoder(CE_MODEL_NAME)
    chunker  = SemanticChunker(encoder)
    retriever = HybridRetriever(encoder)

    print("\n Değerlendirme çalışıyor...")
    results = []
    for i, s in enumerate(samples, 1):
        r = run_pipeline(s, chunker, retriever, ce_model, encoder)
        results.append(r)
        if i % 10 == 0:
            print(f"   {i}/{len(samples)} tamamlandı")

    agg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    print("\n" + "="*50)
    print(f"{'Metrik':<25} {'Skor':>10}")
    print("="*50)
    labels = {'faithfulness': 'Sadakat', 'context_precision': 'Bağlam Hassasiyeti',
              'answer_relevancy': 'Yanıt Alaka Düzeyi', 'token_count': 'Token/Sorgu'}
    for k, v in agg.items():
        print(f"{labels.get(k, k):<25} {v:>10.3f}")
    print("="*50)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'aggregate': agg, 'per_sample': results}, f, indent=2)
    print(f"\n Sonuçlar kaydedildi: {args.output}")


def mode_demo(args):
    """Tek sorgu demo modu."""
    print(f"\n Sorgu: '{args.query}'")
    print("🔧 Modeller yükleniyor...")
    encoder   = SentenceTransformer(ENCODER_MODEL)
    ce_model  = CrossEncoder(CE_MODEL_NAME)
    chunker   = SemanticChunker(encoder)
    retriever = HybridRetriever(encoder)

    # Demo bağlamı
    sample = {
        'question': args.query,
        'context': (
            "William Shakespeare wrote Hamlet around 1600-1601. "
            "It is one of his most famous tragedies. "
            "The play is set in Denmark and follows Prince Hamlet. "
            "Shakespeare was born in Stratford-upon-Avon in 1564. "
            "He wrote 37 plays in total during his lifetime."
        ),
        'answer': 'Shakespeare'
    }

    chunks = chunker.chunk(sample['context'])
    retriever.build(chunks)
    cands  = retriever.search(args.query, k=5)
    ranked = rerank(args.query, cands, ce_model, top_k=3)
    final  = compress(args.query, ranked, encoder, ratio=0.6)

    print(f"\n Sıkıştırılmış Bağlam:\n   {final[0]}")

    prompt = (
        f"Bağlam bilgisine dayanarak soruyu kısa yanıtla.\n\n"
        f"Bağlam: {final[0]}\n\n"
        f"Soru: {args.query}\n\nCevap:"
    )
    print(f"\n Oluşturulan Prompt:\n{prompt}")


def mode_ablation(args):
    """Ablation modu — özet tablo."""
    print(" Ablasyon modu — tam analiz için notebook kullanın.")
    print("   Notebook: notebooks/02_IntelliRAG_Ablation.ipynb")


# ── Ana giriş ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='IntelliRAG Değerlendirme Betiği',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--mode',      choices=['evaluate', 'ablation', 'demo'], default='evaluate')
    parser.add_argument('--n_samples', type=int, default=50, help='Değerlendirilecek örnek sayısı')
    parser.add_argument('--output',    default='results/metrics.json', help='Çıktı JSON dosyası')
    parser.add_argument('--query',     default='Who wrote Hamlet?', help='Demo modu sorgusu')

    args = parser.parse_args()

    print(f" IntelliRAG Değerlendirme — mod: {args.mode}")
    print(f"   Cihaz: {DEVICE}\n")

    if args.mode == 'evaluate':
        mode_evaluate(args)
    elif args.mode == 'demo':
        mode_demo(args)
    elif args.mode == 'ablation':
        mode_ablation(args)


if __name__ == '__main__':
    main()
