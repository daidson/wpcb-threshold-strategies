def plot_training_curves(history: list[dict], output_path: str) -> None:
    import matplotlib.pyplot as plt

    iters = [h["iteration"] for h in history]
    map50 = [h["metrics"].box.map50 for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(iters, map50, marker="o")
    plt.xlabel("Self-training iteration")
    plt.ylabel("mAP@50")
    plt.title("Self-training progress")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
