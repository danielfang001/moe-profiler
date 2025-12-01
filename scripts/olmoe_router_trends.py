import os
import glob
import torch
import numpy as np
from safetensors import safe_open
import matplotlib.pyplot as plt

NUM_RUNS = 1000

def load_olmoe_router_weights(shard_dir):
    """
    Extract router weight matrices from OLMoE.
    OLMoE routers live at: model.layers.L.mlp.gate.weight
    """
    shards = sorted(glob.glob(os.path.join(shard_dir, "model-*.safetensors")))
    gate_weights = {}

    for shard in shards:
        print(f"Scanning shard: {shard}")
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                # True router for OLMoE
                if key.endswith("mlp.gate.weight"):
                    print("   Found router:", key)
                    gate_weights[key] = f.get_tensor(key)

    return gate_weights

def calculate_geometric_elbow(y):
    """
    Geometric elbow: find point farthest from diagonal in normalized space.
    Assumes OLMoE's 64 experts.
    """
    y = np.asarray(y)
    n = len(y)
    assert n == 64, f"Expected 64 experts, got {n}"

    x_norm = np.linspace(0, 1, n)

    # Normalize y so that y[0] = 0, y[-1] = 1
    y_norm = (y[0] - y) / (y[0] - y[-1] + 1e-12)

    diff = y_norm - x_norm
    elbow_idx = int(np.argmax(diff))
    return elbow_idx


def elbow_angle(y, elbowidx):
    """
    Returns elbow angle in degrees.
    Smaller angle → sharper elbow → more sparsity.
    """
    y_np = np.asarray(y)
    n = len(y_np)
    x_norm = np.linspace(0, 1, n)

    if elbowidx == 0 or elbowidx == n - 1:
        return 180.0

    y_range = y_np[0] - y_np[-1]
    if y_range == 0:
        return 180.0

    y_norm = (y_np[0] - y_np) / (y_range + 1e-12)

    pe = np.array([x_norm[elbowidx], y_norm[elbowidx]])
    ps = np.array([0.0, 0.0])
    pend = np.array([1.0, 1.0])

    v1 = ps - pe
    v2 = pend - pe

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))

def compute_layer_metrics(gate_weights, num_runs=NUM_RUNS):
    """
    Compute aggregated metrics across multiple random probes.
    """
    metrics = []

    for key, W in gate_weights.items():
        layer = int(key.split(".")[2])
        W = W.float()
        hidden_dim = W.shape[1]

        entropy_vals = []
        elbow_vals = []
        spec_vals = []
        normvar_vals = []

        # Precompute expert geometry
        norms = torch.norm(W, dim=1).cpu().numpy()
        norm_var = norms.var()

        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        sim = (Wn @ Wn.T).cpu().numpy()
        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        specialization = 1 - upper.mean()

        # Run multiple Monte Carlo probes
        for _ in range(num_runs):
            hidden = torch.randn(32, hidden_dim)
            logits = (hidden @ W.T).detach().cpu().numpy()

            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)

            sorted_probs = np.sort(probs, axis=1)[:, ::-1].mean(axis=0)

            elbow_idx = calculate_geometric_elbow(sorted_probs)
            elbow_ang = elbow_angle(sorted_probs, elbow_idx)
            entropy = -(sorted_probs * np.log(sorted_probs + 1e-12)).sum()

            entropy_vals.append(entropy)
            elbow_vals.append(elbow_ang)

        # Record mean and std
        metrics.append({
            "layer": layer,
            "entropy_mean": np.mean(entropy_vals),
            "entropy_std": np.std(entropy_vals),

            "elbow_angle_mean": np.mean(elbow_vals),
            "elbow_angle_std": np.std(elbow_vals),

            "specialization": specialization,  # geometry-based, not MC
            "norm_variance": norm_var
        })

    return sorted(metrics, key=lambda x: x["layer"])

def plot_router_trends(metrics):
    layers = [m["layer"] for m in metrics]

    plt.figure(figsize=(12, 8))

    # Elbow Angle
    plt.subplot(2, 2, 1)
    means = [m["elbow_angle_mean"] for m in metrics]
    stds = [m["elbow_angle_std"] for m in metrics]
    plt.plot(layers, means)
    plt.fill_between(layers, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    plt.title("OLMoE Router Elbow Angle vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Angle (degrees)")

    # Entropy
    plt.subplot(2, 2, 2)
    means = [m["entropy_mean"] for m in metrics]
    stds = [m["entropy_std"] for m in metrics]
    plt.plot(layers, means)
    plt.fill_between(layers, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    plt.title("Router Entropy vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Entropy")

    # Specialization
    plt.subplot(2, 2, 3)
    plt.plot(layers, [m["specialization"] for m in metrics])
    plt.title("Expert Specialization vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("1 - Mean Cosine Similarity")

    # Norm Variance
    plt.subplot(2, 2, 4)
    plt.plot(layers, [m["norm_variance"] for m in metrics])
    plt.title("Expert Norm Variance vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Var(||W||)")

    plt.tight_layout()
    plt.show()

def main():
    script_dir = os.path.dirname(__file__)
    shard_dir = os.path.join(script_dir, "..", "olmoe")

    print("Loading OLMoE router weights from:", shard_dir)
    gate_weights = load_olmoe_router_weights(shard_dir)

    print(f"\nLoaded {len(gate_weights)} OLMoE router matrices.")
    metrics = compute_layer_metrics(gate_weights)

    print("Plotting trends...")
    plot_router_trends(metrics)


if __name__ == "__main__":
    main()
