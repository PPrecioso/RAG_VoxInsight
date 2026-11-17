"""Índice RAG local: embeddings LSA ou SentenceTransformers opcional."""
from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def create_index(documents: pd.DataFrame, output: str | Path, model_name: str | None = None, backend: str = 'lite') -> None:
    """Builds dense LSA embeddings (lite) or neural embeddings (transformer)."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    texts = documents['text'].fillna('').astype(str).tolist()
    if not texts:
        raise ValueError('Nenhuma avaliação com texto encontrada.')
    print('Criando índice lexical TF-IDF...', flush=True)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents='unicode', ngram_range=(1, 2), min_df=1, max_features=30000, dtype=np.float32)
    lexical = vectorizer.fit_transform(texts)
    joblib.dump((vectorizer, lexical), output / 'lexical.joblib', compress=3)
    if backend == 'lite':
        print('Gerando embeddings densos LSA, sem PyTorch...', flush=True)
        dimensions = min(96, lexical.shape[0]-1, lexical.shape[1]-1)
        if dimensions < 1:
            raise ValueError('São necessárias pelo menos duas avaliações e duas palavras distintas.')
        svd = TruncatedSVD(n_components=dimensions, n_iter=3, random_state=42)
        emb = normalize(svd.fit_transform(lexical)).astype(np.float32)
        joblib.dump(svd, output / 'svd.joblib', compress=3)
        config = {'backend': 'lite', 'dimensions': dimensions, 'documents': len(texts)}
    elif backend == 'transformer':
        model_name = model_name or os.getenv('EMBEDDING_MODEL', DEFAULT_MODEL)
        encoder = load_encoder(model_name)
        emb = np.asarray(encoder.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)
        config = {'backend': 'transformer', 'embedding_model': model_name, 'documents': len(texts)}
    else:
        raise ValueError('Backend deve ser lite ou transformer')
    np.save(output / 'embeddings.npy', emb)
    documents.to_pickle(output / 'documents.pkl')
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Índice pronto: {len(texts)} avaliações; {emb.shape[1]} dimensões.', flush=True)


class ReviewIndex:
    def __init__(self, location: str | Path, encoder=None):
        location = Path(location)
        if not (location / 'config.json').exists():
            raise FileNotFoundError('Índice ausente. Execute: python -m voc_rag.cli build')
        conf = json.loads((location / 'config.json').read_text(encoding='utf-8'))
        self.df = pd.read_pickle(location / 'documents.pkl')
        self.emb = np.load(location / 'embeddings.npy', mmap_mode='r')
        self.vectorizer, self.lexical = joblib.load(location / 'lexical.joblib')
        self.backend = conf.get('backend', 'transformer')
        if self.backend == 'lite':
            self.svd = joblib.load(location / 'svd.joblib')
            self.encoder = None
        else:
            self.encoder = encoder or load_encoder(conf['embedding_model'])
        self.model_name = conf.get('embedding_model', 'LSA / scikit-learn')

    def retrieve(self, question: str, k: int = 8, min_score: int | None = None,
                 max_score: int | None = None, category: str | None = None) -> list[dict]:
        q = question.strip()
        if len(q) < 4:
            return []
        valid = np.ones(len(self.df), dtype=bool)
        if min_score is not None:
            valid &= self.df['review_score'].to_numpy() >= min_score
        if max_score is not None:
            valid &= self.df['review_score'].to_numpy() <= max_score
        if category:
            valid &= self.df['categories'].fillna('').str.contains(category, regex=False, case=False).to_numpy()
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            return []
        q_lex = self.vectorizer.transform([q])
        if q_lex.nnz == 0:
            return []
        if self.backend == 'lite':
            q_emb = normalize(self.svd.transform(q_lex)).astype(np.float32)[0]
        else:
            q_emb = np.asarray(self.encoder.encode([q], normalize_embeddings=True), dtype=np.float32)[0]
        semantic = np.asarray(self.emb[idx] @ q_emb).reshape(-1)
        lexical = np.asarray((self.lexical[idx] @ q_lex.T).toarray()).ravel()
        hybrid = 0.65 * semantic + 0.35 * lexical if self.backend == 'lite' else 0.80 * semantic + 0.20 * lexical
        top = np.argsort(-hybrid)[:min(k, len(idx))]
        results = []
        for pos in top:
            row = self.df.iloc[int(idx[pos])].to_dict()
            row.update({'similarity': float(semantic[pos]), 'lexical': float(lexical[pos]), 'relevance': float(hybrid[pos])})
            results.append(row)
        return results
