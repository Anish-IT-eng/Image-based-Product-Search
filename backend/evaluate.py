"""
evaluate.py
-----------
Measures retrieval accuracy of the current FAISS index.

Metric: Recall@K
  For each query image in the catalog, the relevant set = all other images
  from the same Brand+Subcategory (same shoe type, same brand).
  Recall@K = fraction of queries where at least 1 relevant item appears in
             the top-K retrieved results (excluding the query itself).

Usage:
    python backend/evaluate.py --sample 200

Prints:
    Recall@1, Recall@5, Recall@10, Recall@20
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)


def load_index_and_catalog():
    from indexer import FaissIndex
    from config import settings
    idx = FaissIndex()
    idx.load()
    return idx


def compute_recall(idx, sample: int = 200, ks=(1, 5, 10, 20), seed: int = 42):
    catalog = idx.catalog
    n = len(catalog)

    # Build brand+subcategory groups
    # "Relevant" = same brand AND same subcategory (same specific shoe type)
    from collections import defaultdict
    group_map = defaultdict(list)
    for i, entry in enumerate(catalog):
        brand = (entry.get("brand") or "").strip()
        subcat = (entry.get("subcategory") or "").strip()
        key = f"{brand}||{subcat}"
        group_map[key].append(i)

    # Filter: only queries that have at least 1 other item in same group
    valid_queries = [
        i for i, entry in enumerate(catalog)
        if len(group_map[f"{(entry.get('brand') or '').strip()}||{(entry.get('subcategory') or '').strip()}"]) > 1
    ]

    if not valid_queries:
        print("ERROR: No valid queries found — no brand+subcategory group has >1 image.")
        print("Falling back to brand-only grouping...")
        group_map = defaultdict(list)
        for i, entry in enumerate(catalog):
            brand = (entry.get("brand") or "").strip()
            group_map[brand].append(i)
        valid_queries = [
            i for i in range(n)
            if len(group_map[(catalog[i].get("brand") or "").strip()]) > 1
        ]

    rng = random.Random(seed)
    if len(valid_queries) > sample:
        query_indices = rng.sample(valid_queries, sample)
    else:
        query_indices = valid_queries

    print(f"  Evaluating {len(query_indices)} queries from {n} total catalog items")
    print(f"  Groups (brand+subcategory) with >1 image: {sum(1 for g in group_map.values() if len(g)>1)}")
    print()

    max_k = max(ks)
    recall_counts = {k: 0 for k in ks}

    # Load embeddings from the saved .npy file (avoids FAISS reconstruct complexity)
    from config import settings
    emb_path = settings.data_dir / "embeddings.npy"
    if emb_path.exists():
        all_embeddings = np.load(str(emb_path)).astype(np.float32)
        # Handle case where npy has more/fewer rows than current catalog
        if len(all_embeddings) != n:
            print(f"  WARNING: embeddings.npy has {len(all_embeddings)} rows but catalog has {n}. Truncating.")
            all_embeddings = all_embeddings[:n]
    else:
        print(f"  ERROR: embeddings.npy not found at {emb_path}")
        print("  Run build_index.py first.")
        sys.exit(1)

    for qi in query_indices:
        entry = catalog[qi]
        brand = (entry.get("brand") or "").strip()
        subcat = (entry.get("subcategory") or "").strip()
        key = f"{brand}||{subcat}"
        relevant_set = set(group_map[key]) - {qi}

        if not relevant_set:
            continue

        # Query with max_k+1 to exclude self
        q_vec = all_embeddings[qi:qi+1]
        k_search = min(max_k + 1, n)
        scores, indices = idx.index.search(q_vec, k_search)

        # Remove self from results
        retrieved = [int(idx_) for idx_ in indices[0] if idx_ != qi and idx_ != -1]

        for k in ks:
            top_k_set = set(retrieved[:k])
            if top_k_set & relevant_set:
                recall_counts[k] += 1

    total = len(query_indices)
    return {k: recall_counts[k] / total for k in ks}, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200, help="Number of query images to evaluate")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  VisualFind — Retrieval Evaluation")
    print("=" * 60 + "\n")

    print("Loading FAISS index and catalog...")
    idx = load_index_and_catalog()
    print(f"  Index: {idx.index.ntotal} vectors, dim={idx.index.d}")
    print(f"  Catalog: {len(idx.catalog)} products\n")

    print("Computing Recall@K...")
    recalls, total = compute_recall(idx, sample=args.sample)

    print("=" * 60)
    print(f"  Results over {total} queries:")
    for k, v in recalls.items():
        print(f"  Recall@{k:2d} = {v*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
