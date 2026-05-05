import json
import subprocess
import sys
from pathlib import Path


COMBOS = [
    {"name": "w_rho", "weight_mode": "rho"},
    {"name": "w_rho_eta", "weight_mode": "rho_eta"},
]


def main():
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path("/tmp/fedmfg_brat500_head_weight_ablation_20260505")
    output_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        "train.py",
        "--root_dir", "/data/zikun_workspace/preprocessed",
        "--seed", "42",
        "--client_names", "BraTS", "Shanghai", "Figshare", "Brisc2025",
        "--global_rounds", "10",
        "--eval_gap", "1",
        "--local_epochs", "3",
        "--local_learning_rate", "1e-3",
        "--client_batch_size_map", "BraTS=8", "Shanghai=64", "Figshare=128", "Brisc2025=128",
        "--client_train_max_samples_map", "BraTS=500",
        "--val_ratio", "0.1",
        "--model_name", "resnet18",
        "--model_mode", "multimodal",
        "--num_classes", "5",
        "--prototype_dim", "512",
        "--dropout", "0.0",
        "--save_gap", "0",
        "--save_total_limit", "1",
        "--save_dir", "placeholder",
        "--algo", "fedmfg",
        "--client_gpu_map", "BraTS=4,5,6,7", "Shanghai=2,3", "Figshare=0", "Brisc2025=1",
        "--mfg_proto_lambda", "0.1",
        "--mfg_head_lambda", "0.1",
        "--mfg_proto_momentum", "0.5",
        "--mfg_proto_tau", "1.0",
        "--mfg_teacher_lambda", "0.5",
        "--mfg_teacher_tau", "1.0",
        "--mfg_head_tau", "1.0",
        "--mfg_head_beta", "1.0",
        "--server_early_stopping_patience", "10",
        "--history_flush_gap_steps", "0",
        "--plot_refresh_gap_steps", "0",
        "--num_workers", "32",
    ]

    results = []
    for combo in COMBOS:
        run_dir = output_root / combo["name"]
        ckpt_dir = run_dir / "ckpt"
        plot_dir = run_dir / "plots"
        history_path = run_dir / "history.json"
        log_path = run_dir / "train.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)

        cmd = list(base_cmd)
        save_dir_index = cmd.index("--save_dir") + 1
        cmd[save_dir_index] = str(ckpt_dir)
        cmd.extend(
            [
                "--history_path", str(history_path),
                "--plot_dir", str(plot_dir),
                "--mfg_head_weight_mode", combo["weight_mode"],
            ]
        )

        print(f"START {combo['name']} weight_mode={combo['weight_mode']}", flush=True)
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            results.append(
                {
                    "name": combo["name"],
                    "weight_mode": combo["weight_mode"],
                    "status": "failed",
                    "returncode": proc.returncode,
                    "log_path": str(log_path),
                }
            )
            print(f"FAILED {combo['name']} rc={proc.returncode}", flush=True)
            continue

        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)

        item = {
            "name": combo["name"],
            "weight_mode": combo["weight_mode"],
            "status": "ok",
            "best_round": history.get("best_round"),
            "best_val_accuracy": history.get("best_val_accuracy"),
            "final_val_accuracy": history.get("val_accuracy", [None])[-1],
            "final_val_loss": history.get("val_loss", [None])[-1],
            "log_path": str(log_path),
            "history_path": str(history_path),
        }
        results.append(item)
        print(
            f"DONE {combo['name']} best={item['best_val_accuracy']:.6f} final={item['final_val_accuracy']:.6f}",
            flush=True,
        )

    ranking = [
        item for item in results
        if item.get("status") == "ok" and item.get("best_val_accuracy") is not None
    ]
    ranking.sort(key=lambda item: item["best_val_accuracy"], reverse=True)

    summary = {
        "config": {
            "global_rounds": 10,
            "local_epochs": 1,
            "brats_train_max_samples": 500,
            "other_clients": "full train split",
            "mfg_proto_lambda": 0.1,
            "mfg_head_lambda": 0.1,
            "head_weight_modes": ["rho", "rho_eta"],
            "client_gpu_map": {
                "BraTS": [4, 5, 6, 7],
                "Shanghai": [2, 3],
                "Figshare": [0],
                "Brisc2025": [1],
            },
        },
        "results": results,
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)
    if ranking:
        best = ranking[0]
        print(
            f"BEST {best['name']} weight_mode={best['weight_mode']} best_val={best['best_val_accuracy']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
