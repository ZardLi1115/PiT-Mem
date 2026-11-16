"""GPU / CPU batch embedding pipeline with sharded idempotent checkpointing for FinPiT-Bench."""

from __future__ import annotations

import os
import json
from typing import List, Optional

import numpy as np
import pandas as pd


def compute_embeddings_sharded(
    shard_parquets: List[str],
    output_dir: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    dim: int = 768,
    batch_size: int = 256,
    use_gpu_model: bool = True,
) -> List[str]:
    """
    Encode text shards into normalized float32/float16 embeddings with atomic shard checkpointing.
    On Colab Pro (L4/A100), uses SentenceTransformers / HuggingFace FP16 (`BAAI/bge-base-en-v1.5` or `BAAI/bge-m3`).
    Falls back cleanly to deterministic multi-gram semantic random-projection if running in offline CPU CI.
    """
    os.makedirs(output_dir, exist_ok=True)
    vec_paths: List[str] = []

    encoder = None
    if use_gpu_model:
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder = SentenceTransformer(model_name, device=device)
            encoder.max_seq_length = 96
            if device == "cuda":
                encoder.half()
            print(f"[Embed] Loaded {model_name} on {device} (batch_size={batch_size}).")
        except Exception as exc:
            print(f"[Embed fallback] Transformer model unavailable ({exc}); using deterministic n-gram projection.")
            encoder = None

    for idx, pq_path in enumerate(shard_parquets):
        out_npy = os.path.join(output_dir, f"vecs_shard_{idx:04d}.npy")
        vec_paths.append(out_npy)
        if os.path.exists(out_npy):
            continue  # Resume from checkpoint!

        df = pd.read_parquet(pq_path, columns=["text"])
        texts = df["text"].astype(str).tolist()

        if encoder is not None:
            embs = encoder.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            if embs.shape[1] > dim:
                embs = embs[:, :dim].copy()
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs /= norms
        else:
            from sklearn.feature_extraction.text import HashingVectorizer

            hv = HashingVectorizer(n_features=dim, alternate_sign=True, norm="l2", ngram_range=(1, 2))
            mat = hv.transform(texts)
            embs = mat.toarray().astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            zero_mask = (norms[:, 0] < 1e-8)
            if zero_mask.any():
                rng = np.random.default_rng(1000 + idx)
                embs[zero_mask] = rng.standard_normal((int(zero_mask.sum()), dim)).astype(np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs /= (norms + 1e-12)

        tmp_npy = out_npy + ".tmp.npy"
        np.save(tmp_npy, embs)
        os.replace(tmp_npy, out_npy)
        print(f"[Embed] Saved shard {idx + 1}/{len(shard_parquets)} ({len(embs):,} vectors) -> {os.path.basename(out_npy)}", flush=True)

    if encoder is not None:
        del encoder
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return vec_paths
