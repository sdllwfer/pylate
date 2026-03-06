"""
ColBERT 推理脚本（使用 pylate的rank.rerank 接口）
"""

import argparse
import numpy as np
import torch
from pylate import models, rank


def load_model(model_path: str):
    model = models.ColBERT(model_name_or_path=model_path)
    return model


def search(model, query: str, documents: list[str], top_k: int = 5):
    # 编码查询
    queries_list = model.encode(
        [query],
        is_query=True,
        show_progress_bar=False,
    )

    # 编码文档 - 每个文档单独编码为 token embeddings
    documents_list = model.encode(
        documents,
        is_query=False,
        show_progress_bar=False,
    )

    # 调试
    print(f"   queries[0] shape: {queries_list[0].shape}")
    print(f"   documents[0] shape: {documents_list[0].shape}")

    # rank.rerank 期望格式：
    # documents_ids: [[id1, id2, ...]] - 每个查询对应的文档ID列表
    # documents_embeddings: [[tensor1, tensor2, ...]] - 每个查询对应的文档embeddings列表
    documents_ids = [[str(i) for i in range(len(documents))]]
    documents_embeddings = [documents_list]  # 包装成嵌套列表

    # 使用 rerank 计算分数
    reranked = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=queries_list,
        documents_embeddings=documents_embeddings,
    )

    results = []
    for query_results in reranked:
        for doc in query_results:
            doc_id = int(doc["id"])
            score = doc["score"]
            results.append((documents[doc_id], score, doc_id))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description='ColBERT 推理脚本')
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--query', type=str, default='What is machine learning?')
    parser.add_argument('--documents', type=str, nargs='+', default=None)
    parser.add_argument('--top_k', type=int, default=5)

    args = parser.parse_args()

    print(f"📥 加载模型: {args.model_path}")
    model = load_model(args.model_path)
    print("✅ 模型加载成功!\n")

    if args.documents is None:
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning is a subset of machine learning based on artificial neural networks.",
            "Python is a high-level programming language.",
            "Natural language processing is a subfield of AI.",
            "Computer vision deals with how computers can gain understanding from images.",
            "Reinforcement learning is an area of machine learning.",
            "Data science is a field that uses scientific methods to extract insights.",
        ]
    else:
        documents = args.documents

    print(f"🔍 查询: \"{args.query}\"\n")
    print(f"📚 文档库: {len(documents)} 个文档\n")

    results = search(model, args.query, documents, top_k=args.top_k)

    print("=" * 60)
    print("📊 检索结果 (Top {}):".format(args.top_k))
    print("=" * 60)

    for i, (doc, score, doc_id) in enumerate(results, 1):
        print(f"\n{i}. [Score: {score:.4f}] [ID: {doc_id}]")
        print(f"   {doc[:80]}..." if len(doc) > 80 else f"   {doc}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
