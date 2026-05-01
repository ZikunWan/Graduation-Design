import json
import logging
import os
from pathlib import Path

import torch

from train import ALGO_MAP, build_arg_parser


def get_args():
    parser = build_arg_parser()
    parser.description = "Evaluate saved checkpoints on the test split"
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path for federated algorithms. "
            "For algo=local, this can be a save_dir or any *_best.pth inside that directory."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save the test summary as JSON.",
    )
    return parser.parse_args()


def _clone_to_jsonable(value):
    if isinstance(value, dict):
        return {key: _clone_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_clone_to_jsonable(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _resolve_federated_checkpoint(args):
    checkpoint_path = args.checkpoint or os.path.join(args.save_dir, "best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_local_checkpoint_dir(args):
    checkpoint_input = args.checkpoint or args.save_dir
    checkpoint_path = Path(checkpoint_input)
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    return checkpoint_dir


def _load_local_best_checkpoints(server, checkpoint_dir):
    missing_clients = []
    for client in server.clients:
        best_path = checkpoint_dir / f"{client.client_name}_best.pth"
        if not best_path.exists():
            missing_clients.append(client.client_name)
            continue

        checkpoint = torch.load(best_path, map_location="cpu")
        client_state = checkpoint.get("client_state")
        if client_state is None:
            raise ValueError(f"Local best checkpoint missing client_state: {best_path}")
        client.load_state(client_state)
        client.best_round = checkpoint.get("best_round")
        client.best_val_accuracy = checkpoint.get("best_val_accuracy")

    if missing_clients:
        raise FileNotFoundError(
            f"Missing local best checkpoints for clients: {', '.join(missing_clients)}"
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("test")

    args = get_args()
    server_cls = ALGO_MAP[args.algo]
    server = server_cls(args)

    if args.algo == "local":
        checkpoint_dir = _resolve_local_checkpoint_dir(args)
        _load_local_best_checkpoints(server, checkpoint_dir)
        logger.info("Loaded local best checkpoints from %s", checkpoint_dir)
        round_idx = max(
            (
                client.best_round
                for client in server.clients
                if getattr(client, "best_round", None) is not None
            ),
            default=0,
        )
    else:
        checkpoint_path = _resolve_federated_checkpoint(args)
        round_idx = server.load_checkpoint(checkpoint_path)
        logger.info("Loaded checkpoint from %s", checkpoint_path)

    summary = server.evaluate(split="test", round_idx=round_idx)
    summary["round"] = round_idx

    logger.info("─" * 60)
    logger.info("Test results (round %d)", summary["round"])
    logger.info("  Avg Accuracy : %.4f  ±  %.4f", summary["accuracy"], summary["std_accuracy"])
    logger.info("  Avg Macro-F1 : %.4f  ±  %.4f", summary["macro_f1"], summary["std_macro_f1"])
    logger.info("  Avg Loss     : %.4f", summary["loss"])
    logger.info("─" * 60)
    logger.info("Per-client breakdown:")
    for item in summary["per_client_results"]:
        logger.info(
            "  %-12s  acc=%.4f  f1=%.4f  loss=%.4f  (n=%d)",
            item["client_name"],
            item["accuracy"],
            item["macro_f1"],
            item["loss"],
            item["num_samples"],
        )

    if args.output_json:
        output_path = Path(args.output_json)
        if output_path.parent:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(_clone_to_jsonable(summary), f, indent=2)
        logger.info("Saved test summary to %s", output_path)


if __name__ == "__main__":
    main()
