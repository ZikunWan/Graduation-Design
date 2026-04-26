import numpy as np
import torch
from sklearn.cluster import KMeans


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_per_modality_class_prototypes(
    modality_prototypes,
    labels,
    modality_mask,
    modality_order,
    num_prototypes,
    random_state=0,
):
    modality_prototypes_np = to_numpy(modality_prototypes)
    labels_np = to_numpy(labels).astype(np.int64)
    modality_mask_np = to_numpy(modality_mask).astype(np.float32)
    class_ids = sorted(np.unique(labels_np).tolist())

    prototypes = {}
    for modality_index, modality in enumerate(modality_order):
        prototypes[modality] = {}
        active_index = modality_mask_np[:, modality_index] > 0

        for class_id in class_ids:
            class_index = labels_np == class_id
            selected = modality_prototypes_np[active_index & class_index, modality_index]
            if len(selected) == 0:
                continue

            num_clusters = min(num_prototypes, len(selected))
            if num_clusters == 1:
                centers = selected.mean(axis=0, keepdims=True)
                counts = np.array([len(selected)], dtype=np.float32)
            else:
                kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init="auto")
                cluster_ids = kmeans.fit_predict(selected)
                centers = kmeans.cluster_centers_
                counts = np.array([(cluster_ids == cluster_id).sum() for cluster_id in range(num_clusters)], dtype=np.float32)

            prototypes[modality][int(class_id)] = {
                "centers": torch.tensor(centers, dtype=torch.float32),
                "counts": torch.tensor(counts, dtype=torch.float32),
            }

    return prototypes


def build_prototype_tensor(
    prototypes,
    modality_order,
    num_classes,
    num_prototypes,
    feature_dim,
):
    prototype_tensor = torch.zeros(
        len(modality_order),
        num_classes,
        num_prototypes,
        feature_dim,
        dtype=torch.float32,
    )
    prototype_mask = torch.zeros(
        len(modality_order),
        num_classes,
        num_prototypes,
        dtype=torch.float32,
    )
    prototype_count = torch.zeros(
        len(modality_order),
        num_classes,
        num_prototypes,
        dtype=torch.float32,
    )

    for modality_index, modality in enumerate(modality_order):
        for class_id, value in prototypes[modality].items():
            centers = value["centers"]
            counts = value["counts"]
            count = min(num_prototypes, centers.shape[0])
            prototype_tensor[modality_index, class_id, :count] = centers[:count]
            prototype_mask[modality_index, class_id, :count] = 1.0
            prototype_count[modality_index, class_id, :count] = counts[:count]

    return prototype_tensor, prototype_mask, prototype_count


def aggregate_global_prototypes(
    prototype_tensors,
    prototype_masks,
    prototype_counts,
):
    prototype_tensors = torch.stack(prototype_tensors, dim=0)
    prototype_masks = torch.stack(prototype_masks, dim=0)
    prototype_counts = torch.stack(prototype_counts, dim=0)

    weighted_tensors = prototype_tensors * prototype_counts.unsqueeze(-1)
    global_prototype = weighted_tensors.sum(dim=(0, 3))
    global_count = prototype_counts.sum(dim=(0, 3))
    global_mask = (global_count > 0).float()
    global_prototype = global_prototype / global_count.unsqueeze(-1)

    return global_prototype, global_mask, global_count

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(data, path):
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    os.replace(tmp_path, path)


def _apply_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "#f7f5f1",
        "axes.facecolor": "#fffdf9",
        "axes.edgecolor": "#d7d2c8",
        "axes.labelcolor": "#24303a",
        "axes.titlecolor": "#24303a",
        "axes.grid": True,
        "grid.color": "#e3ddd2",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "xtick.color": "#46525c",
        "ytick.color": "#46525c",
        "text.color": "#24303a",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "#f7f5f1",
        "savefig.bbox": "tight",
    })


def _get_palette():
    return [
        "#1f4e5f",
        "#d17b0f",
        "#4f772d",
        "#7c5c48",
        "#7b8fa1",
        "#b56576",
    ]


def _iter_client_histories(history):
    clients = history.get("clients", {})
    return [(name, clients[name]) for name in sorted(clients.keys())]


def _safe_name(name):
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(name))


def _plot_overview_metrics(history, output_dir, algo_name):
    val_acc = history.get("val_accuracy", [])
    val_f1 = history.get("val_macro_f1", [])
    val_loss = history.get("val_loss", [])
    test_acc = history.get("test_accuracy", [])
    test_f1 = history.get("test_macro_f1", [])
    test_loss = history.get("test_loss", [])
    eval_round = history.get("eval_round", [])
    if not any([val_acc, val_f1, val_loss, test_acc, test_f1, test_loss]):
        return

    if eval_round and len(eval_round) == len(val_acc):
        val_x = eval_round
    else:
        val_x = list(range(len(val_acc)))

    final_x = val_x[-1] if val_x else 0
    test_x = [final_x for _ in range(len(test_acc))]

    colors = _get_palette()
    panels = [
        ("Accuracy", val_acc, test_acc, colors[0], "overview_metrics_accuracy.png"),
        ("Macro-F1", val_f1, test_f1, colors[1], "overview_metrics_macro_f1.png"),
        ("Loss", val_loss, test_loss, colors[2], "overview_metrics_loss.png"),
    ]
    for title, val_values, test_values, color, filename in panels:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.5))
        if val_values:
            ax.plot(val_x, val_values, color=color, linewidth=2.2, marker="o", markersize=4, label="Validation")
        if test_values:
            ax.scatter(test_x, test_values, color="#111827", s=48, label="Final Test", zorder=3)
        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Round")
        ax.legend(frameon=False)
        fig.suptitle(f"{algo_name.upper()} Training Overview", fontsize=14, fontweight="semibold")
        fig.savefig(Path(output_dir) / filename, dpi=180)
        plt.close(fig)


def _plot_train_loss(history, output_dir, algo_name):
    client_histories = _iter_client_histories(history)
    available = [(name, item) for name, item in client_histories if item.get("train_step") and item.get("train_loss")]
    if not available:
        return

    colors = _get_palette()
    for index, (client_name, item) in enumerate(available):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
        x = item.get("train_step", [])
        y = item.get("train_loss", [])
        ax.plot(x, y, color=colors[index % len(colors)], linewidth=1.8, label="Loss")

        extra_metrics = item.get("train_metrics", {})
        extra_palette = ["#9c6644", "#6b7280", "#0f766e"]
        for metric_index, metric_name in enumerate(sorted(extra_metrics.keys())[:3]):
            metric_values = extra_metrics[metric_name]
            if len(metric_values) != len(x):
                continue
            ax.plot(
                x,
                metric_values,
                color=extra_palette[metric_index % len(extra_palette)],
                linewidth=1.5,
                linestyle="--",
                alpha=0.9,
                label=metric_name.replace("_", " ").title(),
            )

        ax.set_title(client_name, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        fig.suptitle(f"{algo_name.upper()} Step-wise Training Loss", fontsize=14, fontweight="semibold")
        fig.savefig(Path(output_dir) / f"train_loss_steps_{_safe_name(client_name)}.png", dpi=180)
        plt.close(fig)


def _plot_client_metrics(history, output_dir, algo_name):
    client_histories = _iter_client_histories(history)
    available = []
    for name, item in client_histories:
        if item.get("val_accuracy") or item.get("val_loss") or item.get("test_accuracy") or item.get("final_test_accuracy") is not None:
            available.append((name, item))
    if not available:
        return

    colors = _get_palette()

    for index, (client_name, item) in enumerate(available):
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        val_acc = item.get("val_accuracy", [])
        val_f1 = item.get("val_macro_f1", [])
        val_loss = item.get("val_loss", [])
        val_round = item.get("val_round", [])
        if len(val_round) == len(val_acc) and val_round:
            x = [int(v) for v in val_round]
        else:
            x = list(range(len(val_acc)))
        if val_acc:
            ax.plot(x, val_acc, color=colors[0], linewidth=2.1, marker="o", markersize=3.5, label="Val Acc")
        if val_f1:
            ax.plot(x, val_f1, color=colors[1], linewidth=2.0, marker="o", markersize=3.0, label="Val F1")

        ax2 = ax.twinx()
        if val_loss:
            ax2.plot(x, val_loss, color=colors[2], linewidth=1.8, linestyle="--", label="Val Loss")
        test_acc_series = item.get("test_accuracy", [])
        test_f1_series = item.get("test_macro_f1", [])
        test_loss_series = item.get("test_loss", [])
        test_round_series = item.get("test_round", [])
        final_test_accuracy = item.get("final_test_accuracy")
        final_test_macro_f1 = item.get("final_test_macro_f1")
        final_test_loss = item.get("final_test_loss")
        plot_test_accuracy = final_test_accuracy if final_test_accuracy is not None else (test_acc_series[-1] if test_acc_series else None)
        plot_test_macro_f1 = final_test_macro_f1 if final_test_macro_f1 is not None else (test_f1_series[-1] if test_f1_series else None)
        plot_test_loss = final_test_loss if final_test_loss is not None else (test_loss_series[-1] if test_loss_series else None)
        if test_round_series:
            final_x = int(test_round_series[-1])
        elif x:
            final_x = int(x[-1])
        else:
            final_x = 0
        if plot_test_accuracy is not None:
            ax.scatter([final_x], [plot_test_accuracy], color="#111827", s=42, zorder=4, label="Final Test Acc")
        if plot_test_macro_f1 is not None:
            ax.scatter([final_x], [plot_test_macro_f1], color="#7c3aed", s=42, zorder=4, label="Final Test F1")
        if plot_test_loss is not None:
            ax2.scatter([final_x], [plot_test_loss], color="#374151", s=36, zorder=4, label="Final Test Loss")

        ax.set_title(client_name, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax2.set_ylabel("Loss")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
        fig.suptitle(f"{algo_name.upper()} Client Metrics", fontsize=14, fontweight="semibold")
        fig.savefig(Path(output_dir) / f"client_metrics_{_safe_name(client_name)}.png", dpi=180)
        plt.close(fig)


def save_training_curves(history, output_dir, algo_name="train"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_file in ("overview_metrics.png", "train_loss_steps.png", "client_metrics.png"):
        legacy_path = output_dir / legacy_file
        if legacy_path.exists():
            legacy_path.unlink()
    for pattern in ("overview_metrics_*.png", "train_loss_steps_*.png", "client_metrics_*.png"):
        for path in output_dir.glob(pattern):
            path.unlink()
    _apply_plot_style()
    _plot_overview_metrics(history, output_dir, algo_name)
    _plot_train_loss(history, output_dir, algo_name)
    _plot_client_metrics(history, output_dir, algo_name)
