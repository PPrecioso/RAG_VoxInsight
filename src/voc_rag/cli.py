"""Execute: python -m voc_rag.cli [download|build|ask|stats]."""
from __future__ import annotations
import argparse
from pathlib import Path
from dotenv import load_dotenv
from .data import build_documents, download_dataset
from .index import create_index, ReviewIndex
from .rag import answer_question

ROOT = Path(__file__).resolve().parents[2]


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Voice of Customer Intelligence – Olist RAG")
    parser.add_argument("command", choices=["download", "build", "ask", "stats"])
    parser.add_argument("--data", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--index", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--question", "-q", default="Quais problemas de entrega são relatados?")
    parser.add_argument("--min-score", type=int, choices=range(1, 6))
    parser.add_argument("--max-score", type=int, choices=range(1, 6))
    parser.add_argument("--category", default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--backend", choices=["lite", "transformer"], default="lite", help="lite não precisa de PyTorch")
    parser.add_argument("--provider", choices=["ollama", "openai", "extrativo"], default="ollama")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    if args.command == "download":
        print(f"Dados salvos em: {download_dataset(args.data)}")
    elif args.command == "build":
        df = build_documents(args.data)
        print(f"Indexando {len(df)} avaliações com texto...", flush=True)
        create_index(df, args.index, backend=args.backend)
        print("Índice criado em", args.index)
    elif args.command == "stats":
        df = build_documents(args.data)
        print(f"Documentos indexáveis: {len(df)}")
        print(df["review_score"].value_counts().sort_index().to_string())
        print(f"Categorias conhecidas: {df['categories'].notna().sum()}")
    else:
        idx = ReviewIndex(args.index)
        out = answer_question(idx, args.question, k=args.k, min_score=args.min_score,
                              max_score=args.max_score, category=args.category, provider=args.provider, model=args.model)
        print(out["answer"])
        print("\nEVIDÊNCIAS:")
        for m in out["evidence"]:
            print(f"- [{m['review_id']}] score={m['review_score']} similaridade={m['similarity']:.3f}: {m['text'][:180]}")


if __name__ == "__main__":
    main()
