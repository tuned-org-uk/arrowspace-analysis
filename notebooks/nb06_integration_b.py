"""nb06 · Integration B — Activation-manifold ArrowSpace graph + residual stream trajectory

This script implements the §B section for notebook 06.
It expects the following variables to already exist in the namespace
(populated by running cells §0–§3 of 06__comparative_semantic_probing.ipynb):

    words, labels, X_base, weights, layers, OUTPUT_DIR, KNN_K
    search_elements   (helper defined in §0)
    lambda_full       (from §4 of the notebook)

The FFN bug fix must be applied to §3 before running this script:
    ROLES_ALL = ["W_q", "W_k", "W_v", "W_o", "W_ffn1", "W_ffn2"]
so that act_matrix has shape (200, 36) and act_z is (200, 36).

Outputs
-------
  output__06/act_space_knn_vs_embed_knn.csv
    columns: [word, field, act_lambda, embed_lambda,
              activation_nn_field_purity, embed_nn_field_purity]

Side-effects
------------
  aspace_act  — ArrowSpace object on the full activation manifold
  gl_act      — graph Laplacian (consumed by Integration D, nb08)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from arrowspace import ArrowSpaceBuilder
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# ── §3 FFN bug fix ────────────────────────────────────────────────────────────
# Re-build act_matrix with ALL 6 roles (W_q, W_k, W_v, W_o, W_ffn1, W_ffn2)
# so that act_z is (200, 36) instead of the erroneous (200, 24).

ROLES_ALL = ["W_q", "W_k", "W_v", "W_o", "W_ffn1", "W_ffn2"]
N = len(words)
n_layers = len(layers)
n_roles_all = len(ROLES_ALL)          # 6


def activation_energy(W, x):
    """Scalar activation energy normalised by Frobenius norm."""
    if W.shape[1] != x.shape[0]:
        W = W.T
    proj = W @ x
    return float(np.dot(proj, proj) ** 0.5 / (np.linalg.norm(W, "fro") + 1e-9))


act_matrix_full = np.zeros((N, n_layers * n_roles_all))   # (200, 36)

for n_idx, word_vec in enumerate(X_base):
    col = 0
    for i in range(n_layers):
        for role in ROLES_ALL:
            W = weights[(i, role)]
            act_matrix_full[n_idx, col] = activation_energy(W, word_vec)
            col += 1

print(f"act_matrix_full shape: {act_matrix_full.shape}")
assert act_matrix_full.shape == (200, 36), (
    f"FFN bug fix failed: expected (200, 36), got {act_matrix_full.shape}"
)

# ── §B — z-score and build ArrowSpace graph on activation manifold ────────────

act_z = (act_matrix_full - act_matrix_full.mean(0)) / (act_matrix_full.std(0) + 1e-9)
assert act_z.shape == (200, 36), f"act_z shape mismatch: {act_z.shape}"

GRAPH_PARAMS_ACT = {"eps": 1.9, "k": KNN_K, "topk": 10, "p": 2.0, "sigma": None}

aspace_act, gl_act = (
    ArrowSpaceBuilder()
    .with_seed(42)
    .with_dims_reduction(enabled=False, eps=None)
    .with_sampling("simple", 1.0)
).build_and_store(GRAPH_PARAMS_ACT, act_z.astype(np.float64))

print("ArrowSpace graph built on activation manifold.")

λ_act = search_elements(aspace_act, gl_act, act_z, alpha=0.5)
print(f"λ_act  — min={λ_act.min():.4f}  max={λ_act.max():.4f}  mean={λ_act.mean():.4f}")

# ── k-NN field purity ─────────────────────────────────────────────────────────

def knn_field_purity(X, labels, k=KNN_K):
    """Mean fraction of k-NN neighbours sharing the same semantic field label."""
    D = pairwise_distances(X, metric="cosine")
    np.fill_diagonal(D, np.inf)
    purities = []
    for i in range(len(X)):
        nn_idx = np.argsort(D[i])[:k]
        purity = (labels[nn_idx] == labels[i]).mean()
        purities.append(purity)
    return np.array(purities)


act_purity   = knn_field_purity(act_z,  labels)
embed_purity = knn_field_purity(X_base, labels)

print(f"Mean activation_nn_field_purity : {act_purity.mean():.4f}")
print(f"Mean embed_nn_field_purity      : {embed_purity.mean():.4f}")

if act_purity.mean() > embed_purity.mean():
    print("✓  Activation manifold is MORE semantically coherent than CLS embedding")
else:
    print("△  CLS embedding is more semantically coherent (check graph params)")

# ── Assemble and save output ──────────────────────────────────────────────────

df_b = pd.DataFrame({
    "word":                       words,
    "field":                      labels,
    "act_lambda":                 λ_act,
    "embed_lambda":               lambda_full,   # from §4 of the notebook
    "activation_nn_field_purity": act_purity,
    "embed_nn_field_purity":      embed_purity,
})

output_path = OUTPUT_DIR / "act_space_knn_vs_embed_knn.csv"
df_b.to_csv(output_path, index=False)
print(f"\nSaved → {output_path}")
print(df_b.head(10).to_string(index=False))

# Make notebook-level variables available for Integration D (nb08)
# aspace_act and gl_act are already bound in the calling namespace.
print("\naspace_act and gl_act are available for Integration D (nb08).")
