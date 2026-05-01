import argparse
import logging
import random

import numpy as np
import torch

from model import resolve_model_mode
from server.servergh import ServerFedGH
from server.serverproto import ServerFedProto
from server.serverlg import ServerLGFedAvg
from server.serverlocal import ServerLocal
from server.serveramm import ServerFedAMM
from server.servermm import ServerFedMM
from server.servertgp import ServerFedTGP
from server.serverfd import ServerFD

ALGO_MAP = {
    "fedamm":    ServerFedAMM,
    "fedmm":     ServerFedMM,
    "fedgh":     ServerFedGH,
    "fedproto":  ServerFedProto,
    "lgfedavg":  ServerLGFedAvg,
    "local":     ServerLocal,
    "fedtgp":    ServerFedTGP,
    "fd":        ServerFD,
}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Federated learning trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--algo", type=str, default="fedproto",
                        choices=list(ALGO_MAP.keys()),
                        help="Federated algorithm to run")
    parser.add_argument("--root_dir", type=str, default="data",
                        help="Root directory for preprocessed datasets")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Default device (used in serial mode and for the server)")
    parser.add_argument("--parallel", action="store_true",
                        help="Use one GPU per client (auto-falls back to serial if <2 GPUs)")
    parser.add_argument("--brats_ddp", action="store_true",
                        help="Enable intra-client DDP for BraTS across all algorithms")
    parser.add_argument("--brats_ddp_devices", nargs="+", type=int, default=None,
                        help="CUDA device ids for BraTS intra-client DDP (default: all visible GPUs)")
    parser.add_argument(
        "--client_gpu_map",
        nargs="+",
        type=str,
        default=None,
        help=(
            "[local] Per-client GPU assignment. "
            "Format: ClientName=g0[,g1,...], e.g. "
            "BraTS=1,2,3,4 Shanghai=0 Figshare=0 Brisc2025=0"
        ),
    )
    parser.add_argument(
        "--client_batch_size_map",
        nargs="+",
        type=str,
        default=None,
        help=(
            "[local] Per-client batch size. "
            "Format: ClientName=batch_size, e.g. "
            "BraTS=2 Shanghai=16 Figshare=32 Brisc2025=32"
        ),
    )
    parser.add_argument("--brats_ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend for BraTS intra-client DDP")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--client_names", nargs="+",
                        default=["BraTS", "Shanghai", "Figshare", "Brisc2025"],
                        help="Client names (dataset identifiers)")
    parser.add_argument("--include_yale", action="store_true",
                        help="Include Yale client in training rounds")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap samples per client per split (None = use all). "
                             "Useful for fast iteration during development.")

    parser.add_argument("--global_rounds", type=int, default=50)
    parser.add_argument("--eval_gap",      type=int, default=1,
                        help="Run validation every N rounds during training; test runs once after training")

    parser.add_argument("--save_dir",          type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_total_limit",  type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--resume",            type=str, default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--save_gap",          type=int, default=0,
                        help="Save a resumable round checkpoint every N rounds; <=0 disables periodic checkpoint saving")
    parser.add_argument("--history_path",      type=str, default=None,
                        help="Path to save the live-updating history JSON; defaults to <algo>_history.json")
    parser.add_argument("--plot_dir",          type=str, default=None,
                        help="Directory for auto-refreshed training plots; defaults to plots/<algo>")
    parser.add_argument("--history_flush_gap_steps", type=int, default=10,
                        help="Write history.json every N logged training steps; <=0 disables step-based auto-save")
    parser.add_argument("--plot_refresh_gap_steps", type=int, default=50,
                        help="Refresh training plots every N logged training steps; <=0 disables step-based auto-plot")

    parser.add_argument("--local_epochs",        type=int,   default=1)
    parser.add_argument("--local_learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size",          type=int,   default=8)
    parser.add_argument("--val_ratio",           type=float, default=0.1)
    parser.add_argument("--amp",                 action=argparse.BooleanOptionalAction, default=True,
                        help="Enable CUDA mixed precision for training and evaluation")
    parser.add_argument("--num_workers",         type=int,   default=4,
                        help="Number of DataLoader workers per client")
    parser.add_argument("--pin_memory",          action=argparse.BooleanOptionalAction, default=True,
                        help="Pin host memory for faster host-to-device transfer on CUDA")
    parser.add_argument("--persistent_workers",  action=argparse.BooleanOptionalAction, default=True,
                        help="Keep DataLoader workers alive across epochs when num_workers > 0")
    parser.add_argument("--prefetch_factor",     type=int,   default=2,
                        help="Number of batches prefetched by each DataLoader worker when num_workers > 0")

    parser.add_argument("--model_name",    type=str,   default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--model_mode",    type=str,   default="auto",
                        choices=["auto", "baseline", "multimodal"],
                        help="auto selects the default model for each algorithm; multimodal is kept as an alias for AMM-style models")
    parser.add_argument("--num_classes",   type=int,   default=5)
    parser.add_argument("--prototype_dim", type=int,   default=256)
    parser.add_argument("--dropout",       type=float, default=0.0)

    parser.add_argument("--server_learning_rate", type=float, default=1e-3,
                        help="[FedGH / FedTGP] Server-side learning rate")
    parser.add_argument("--server_epochs",        type=int,   default=5,
                        help="[FedGH / FedTGP] Server-side training epochs per round")
    parser.add_argument("--proto_lambda",         type=float, default=1.0,
                        help="[FedProto / FedTGP] Prototype alignment loss weight")
    parser.add_argument("--fd_lambda",            type=float, default=1.0,
                        help="[FedDF] Feature distillation loss weight")
    parser.add_argument("--amm_mb_lambda",        type=float, default=1.0,
                        help="[FedAMM] Intra-client modality-balance loss weight")
    parser.add_argument("--amm_mc_lambda",        type=float, default=1.0,
                        help="[FedAMM] Inter-client modality-combination balance loss weight")
    parser.add_argument("--fedmm_beta",           type=float, default=0.25,
                        help="[FedMM] Beta for balancing prototype L2 and CE magnitudes")
    parser.add_argument("--fedmm_alpha",          type=float, default=0.05,
                        help="[FedMM] Sharpness of the dynamic lambda transition")
    parser.add_argument("--fedmm_t0",             type=float, default=30.0,
                        help="[FedMM] Transition round from CE-emphasis to prototype-emphasis")
    parser.add_argument("--margin_threshold",     type=float, default=1.0,
                        help="[FedTGP] Margin threshold for prototype separation")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="[Local] Stop a client when validation accuracy does not improve for this many rounds; <=0 disables early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="[Local] Minimum validation-accuracy improvement required to reset early-stopping patience")
    parser.add_argument("--server_early_stopping_patience", type=int, default=5,
                        help="[Federated] Stop training early when aggregated validation accuracy does not improve for this many evaluations; <=0 disables it")
    parser.add_argument("--server_early_stopping_min_delta", type=float, default=0.0,
                        help="[Federated] Minimum aggregated validation-accuracy improvement required to reset early-stopping patience")

    return parser


def get_args():
    return build_arg_parser().parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    args = get_args()
    set_seed(args.seed)

    resolved_model_mode = resolve_model_mode(args.model_mode, args.algo)

    logger.info("Algorithm  : %s", args.algo)
    logger.info("Model mode : %s", resolved_model_mode)
    logger.info("Clients    : %s", args.client_names)
    if args.algo == "local":
        logger.info(
            "Rounds     : %d  |  total local epochs: %d (1 epoch/round)",
            args.global_rounds,
            args.global_rounds,
        )
    else:
        logger.info("Rounds     : %d  |  local epochs: %d", args.global_rounds, args.local_epochs)
    logger.info("Device     : %s  |  parallel: %s", args.device, args.parallel)
    if args.brats_ddp:
        logger.info(
            "BraTS DDP  : enabled  |  backend: %s  |  devices: %s",
            args.brats_ddp_backend,
            "auto" if args.brats_ddp_devices is None else args.brats_ddp_devices,
        )
    if args.client_gpu_map:
        logger.info("Client GPU map: %s", args.client_gpu_map)
    if args.client_batch_size_map:
        logger.info("Client batch-size map: %s", args.client_batch_size_map)
    if args.max_samples is not None:
        logger.warning("DEV MODE: max_samples=%d per client per split", args.max_samples)

    server_cls = ALGO_MAP[args.algo]
    server = server_cls(args)

    if args.resume:
        server.load_checkpoint(args.resume)

    round_summaries = server.train()

    if round_summaries:
        last = round_summaries[-1]
        logger.info("─" * 60)
        logger.info("Final results (round %d)", last["round"])
        logger.info("  Avg Accuracy : %.4f  ±  %.4f", last["accuracy"], last["std_accuracy"])
        logger.info("  Avg Macro-F1 : %.4f  ±  %.4f", last["macro_f1"], last["std_macro_f1"])
        logger.info("  Avg Loss     : %.4f", last["loss"])
        logger.info("─" * 60)
        logger.info("Per-client breakdown:")
        for item in last["per_client_results"]:
            logger.info(
                "  %-12s  acc=%.4f  f1=%.4f  loss=%.4f  (n=%d)",
                item["client_name"], item["accuracy"], item["macro_f1"],
                item["loss"], item["num_samples"],
            )

    server.flush_artifacts(force=True)
    logger.info("Training history saved to %s", server.history_path)
    logger.info("Training plots saved to %s", server.plot_dir)


if __name__ == "__main__":
    main()
