import os
import glob
import torch
import numpy as np
from safetensors import safe_open
import matplotlib.pyplot as plt

NUM_RUNS = 10000  # MONTE-CARLO samples
N_PROBES = 32    # probes per run

def is_mixtral_gate(key):
    return key.endswith("block_sparse_moe.gate.weight")

def is_olmoe_gate(key):
    return key.endswith("mlp.gate.weight")

def load_router_weights(shard_dir, is_gate_fn):
    shards = sorted(glob.glob(os.path.join(shard_dir, "model-*.safetensors")))
    gate_weights = {}

    for shard in shards:
        print(f"Scanning {shard}")
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if is_gate_fn(key):
                    print("   Found router:", key)
                    gate_weights[key] = f.get_tensor(key)

    return gate_weights


# ============================================================
# Geometric elbow & elbow angle
# ============================================================

def calculate_geometric_elbow(y):
    y = np.asarray(y)
    n = len(y)
    x_norm = np.linspace(0, 1, n)
    y_norm = (y[0] - y) / (y[0] - y[-1] + 1e-12)
    diff = y_norm - x_norm
    return int(np.argmax(diff))

def elbow_angle(y, elbowidx):
    y = np.asarray(y)
    n = len(y)
    x_norm = np.linspace(0, 1, n)
    if elbowidx == 0 or elbowidx == n - 1:
        return 180.0

    y_range = y[0] - y[-1]
    y_norm = (y[0] - y) / (y_range + 1e-12)

    pe = np.array([x_norm[elbowidx], y_norm[elbowidx]])
    ps = np.array([0.0, 0.0])
    pend = np.array([1.0, 1.0])

    v1 = ps - pe
    v2 = pend - pe

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.degrees(np.arccos(cos_angle))


# ============================================================
# Compute metrics for one model
# ============================================================

def compute_model_metrics(gate_weights, num_runs=NUM_RUNS):
    results = []

    for key, W in gate_weights.items():
        layer = int(key.split(".")[2])
        W = W.float()
        hidden_dim = W.shape[1]

        # Structural metrics (not Monte Carlo)
        norms = torch.norm(W, dim=1).cpu().numpy()
        norm_variance = norms.var()

        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        sim = (Wn @ Wn.T).cpu().numpy()
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        specialization = 1 - upper.mean()

        entropy_vals, elbow_vals = [], []

        for _ in range(num_runs):
            hidden = torch.randn(N_PROBES, hidden_dim)
            logits = (hidden @ W.T).detach().cpu().numpy()

            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)

            sorted_probs = np.sort(probs, axis=1)[:, ::-1].mean(axis=0)

            elbow_idx = calculate_geometric_elbow(sorted_probs)
            elbow_ang = elbow_angle(sorted_probs, elbow_idx)
            entropy = -(sorted_probs * np.log(sorted_probs + 1e-12)).sum()

            entropy_vals.append(entropy)
            elbow_vals.append(elbow_ang)

        results.append({
            "layer": layer,
            "entropy_mean": np.mean(entropy_vals),
            "entropy_std": np.std(entropy_vals),

            "elbow_mean": np.mean(elbow_vals),
            "elbow_std": np.std(elbow_vals),

            "specialization": specialization,
            "norm_variance": norm_variance,
        })

    return sorted(results, key=lambda x: x["layer"])


# ============================================================
# Compare Mixtral vs OLMoE
# ============================================================

def compare_models(mixtral_metrics, olmoe_metrics):
    mix_layers = [m["layer"] for m in mixtral_metrics]
    olm_layers  = [m["layer"] for m in olmoe_metrics]

    plt.figure(figsize=(14, 10))

    # Elbow angle comparison
    plt.subplot(2, 2, 1)
    plt.plot(mix_layers, [m["elbow_mean"] for m in mixtral_metrics], label="Mixtral")
    plt.plot(olm_layers,  [m["elbow_mean"] for m in olmoe_metrics],  label="OLMoE")
    plt.title("Elbow Angle Comparison")
    plt.ylabel("Angle (degrees)")
    plt.xlabel("Layer")
    plt.legend()

    # Entropy comparison
    plt.subplot(2, 2, 2)
    plt.plot(mix_layers, [m["entropy_mean"] for m in mixtral_metrics], label="Mixtral")
    plt.plot(olm_layers,  [m["entropy_mean"] for m in olmoe_metrics],  label="OLMoE")
    plt.title("Entropy Comparison")
    plt.ylabel("Entropy")
    plt.xlabel("Layer")
    plt.legend()

    # Specialization comparison
    plt.subplot(2, 2, 3)
    plt.plot(mix_layers, [m["specialization"] for m in mixtral_metrics], label="Mixtral")
    plt.plot(olm_layers,  [m["specialization"] for m in olmoe_metrics],  label="OLMoE")
    plt.title("Expert Specialization Comparison")
    plt.ylabel("1 - Mean Cosine Similarity")
    plt.xlabel("Layer")
    plt.legend()

    # Norm variance comparison
    plt.subplot(2, 2, 4)
    plt.plot(mix_layers, [m["norm_variance"] for m in mixtral_metrics], label="Mixtral")
    plt.plot(olm_layers,  [m["norm_variance"] for m in olmoe_metrics],  label="OLMoE")
    plt.title("Norm Variance Comparison")
    plt.ylabel("Var(||W||)")
    plt.xlabel("Layer")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n=== Loading Mixtral ===")
    mixtral_dir = "./mixtral"
    mixtral_gates = load_router_weights(mixtral_dir, is_mixtral_gate)
    mixtral_metrics = compute_model_metrics(mixtral_gates)

    print("\n=== Loading OLMoE ===")
    olmoe_dir = "./olmoe"
    olmoe_gates = load_router_weights(olmoe_dir, is_olmoe_gate)
    olmoe_metrics = compute_model_metrics(olmoe_gates)

    print("\n=== Comparing Models ===")
    compare_models(mixtral_metrics, olmoe_metrics)


if __name__ == "__main__":
    main()
