import os
import glob
import torch
import numpy as np
from safetensors import safe_open
import matplotlib.pyplot as plt

from safetensors import safe_open

with safe_open("model-00001-of-00019.safetensors", framework="pt") as f:
    for key in f.keys():
        if "gate" in key:
            print(key)


# load only gate weights

def load_mixtral_router_weights(shard_dir="."):
    """
    Load ONLY Mixtral router (gate) weight matrices from the safetensor shards.
    """
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

# generate synthetic router logits

def compute_router_logits(gate_weights, hidden_dim=4096, num_tokens=16):
    """
    Simulates router logits by feeding random hidden states through gate W^T * h.
    """
    router_logits = {}

    for name, W in gate_weights.items():
        W = W.float()  # shape: [num_experts, hidden_dim]
        hidden = torch.randn(num_tokens, W.shape[1])  # [N, hidden_dim]
        logits = hidden @ W.T  # [N, num_experts]
        router_logits[name] = logits.detach().cpu().numpy()
    return router_logits

# plot sorted probabilities

def plot_sorted_probs(router_logits):
    for name, logits in router_logits.items():
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        plt.figure(figsize=(7,4))
        for i in range(min(8, probs.shape[0])):
            plt.plot(np.sort(probs[i])[::-1], label=f"Token {i}")
        plt.title(f"Sorted Router Probabilities: {name}")
        plt.xlabel("Expert Rank")
        plt.ylabel("Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    print("Loading Mixtral router (gate) weights from local shards...")
    gate_weights = load_mixtral_router_weights(shard_dir=".")
    print(f"Loaded {len(gate_weights)} gate weight matrices.")

    print("Computing synthetic router logits...")
    router_logits = compute_router_logits(gate_weights)

    print("Plotting sorted probabilities...")
    plot_sorted_probs(router_logits)

if __name__ == "__main__":
    main()
