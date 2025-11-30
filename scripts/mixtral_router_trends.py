# mixtral_router_trends.py
# run with python3 scripts/mixtral_router_trends.py
# Analyze Mixtral router geometry across all layers and plot trends.

import os
import glob
import torch
import numpy as np
from safetensors import safe_open
import matplotlib.pyplot as plt

# -------- STEP 1: Load gate weights --------
def load_mixtral_router_weights(shard_dir="."):
    shards = sorted(glob.glob(os.path.join(shard_dir, "model-*.safetensors")))
    gate_weights = {}

    for shard in shards:
        print(f"Scanning shard: {shard}")
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if "block_sparse_moe.gate" in key and key.endswith("weight"):
                    print(f"  Found gate: {key}")
                    gate_weights[key] = f.get_tensor(key)
    return gate_weights

# Compute router metrics
def compute_layer_metrics(gate_weights):
    metrics = []

    for name, W in gate_weights.items():
        layer = int(name.split(".")[2])  # model.layers.X
        W = W.float()

        # Expert norms
        norms = torch.norm(W, dim=1).cpu().numpy()
        norm_var = norms.var()

        # Expert similarity
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        sim = (Wn @ Wn.T).cpu().numpy()
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        specialization = 1 - upper.mean()

        # Synthetic router logits
        hidden = torch.randn(16, W.shape[1])
        logits = (hidden @ W.T).detach().cpu().numpy()

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        sorted_probs = np.sort(probs, axis=1)[:, ::-1].mean(axis=0)
        elbow_sharpness = sorted_probs[0] - sorted_probs[1]
        entropy = -(sorted_probs * np.log(sorted_probs + 1e-12)).sum()

        metrics.append({
            "layer": layer,
            "entropy": entropy,
            "sharpness": elbow_sharpness,
            "specialization": specialization,
            "norm_variance": norm_var
        })

    metrics = sorted(metrics, key=lambda x: x["layer"])
    return metrics

def plot_router_trends(metrics):
    layers = [m["layer"] for m in metrics]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(layers, [m["sharpness"] for m in metrics])
    plt.title("Elbow Sharpness vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Sharpness")

    plt.subplot(2, 2, 2)
    plt.plot(layers, [m["entropy"] for m in metrics])
    plt.title("Router Entropy vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Entropy")

    plt.subplot(2, 2, 3)
    plt.plot(layers, [m["specialization"] for m in metrics])
    plt.title("Expert Specialization vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("1 - Mean Cosine Similarity")

    plt.subplot(2, 2, 4)
    plt.plot(layers, [m["norm_variance"] for m in metrics])
    plt.title("Expert Norm Variance vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Var(||W_i||)")

    plt.tight_layout()
    plt.show()

def main():
    print("Loading Mixtral router (gate) weights from shards...")
    gate_weights = load_mixtral_router_weights(shard_dir=".")  # loads all available layers from all shards
    print(f"Loaded {len(gate_weights)} gate matrices.")

    print("Computing metrics...")
    metrics = compute_layer_metrics(gate_weights)  # computes metrics for ALL layers present in the shards

    print("Plotting trends...")
    plot_router_trends(metrics)

if __name__ == "__main__":
    main()
