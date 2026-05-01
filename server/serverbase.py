import logging
import concurrent.futures
import os
import glob
import time
import threading
import copy

import numpy as np
import torch

from client.clientbase import Client, _clone_to_cpu
from utils import atomic_write_json, save_training_curves

logger = logging.getLogger(__name__)


class Server:
    """
    Minimal base server for this project.

    The base server keeps only the common infrastructure:
    - client registration
    - weighted metric aggregation
    - round bookkeeping
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.eval_gap = args.eval_gap
        self.client_names = list(args.client_names)
        self.include_yale = args.include_yale
        self.server_early_stopping_patience = getattr(args, "server_early_stopping_patience", 0)
        self.server_early_stopping_min_delta = getattr(args, "server_early_stopping_min_delta", 0.0)
        self.server_early_stopping_enabled = self.server_early_stopping_patience > 0
        self.best_val_accuracy = None
        self.best_round = None
        self.no_improvement_rounds = 0
        self.early_stopped = False
        self.early_stopped_round = None
        self.best_state_snapshot = None

        self.num_clients = len(self.client_names)

        self.clients = []
        self.selected_clients = []
        self.uploaded_client_names = []
        self.uploaded_weights = []

        self.history = {
            "clients": {},
            "val_accuracy": [],
            "val_macro_f1": [],
            "val_loss": [],
            "test_accuracy": [],
            "test_macro_f1": [],
            "test_loss": [],
            "round_elapsed_time": [],
            "eval_round": [],
            "eval_elapsed_time": [],
            "best_val_accuracy": None,
            "best_round": None,
            "early_stopped": False,
            "early_stopped_round": None,
        }

        self.save_dir = getattr(args, "save_dir", "checkpoints")
        self.save_total_limit = getattr(args, "save_total_limit", 3)
        self.save_gap = getattr(args, "save_gap", 0)
        self.saved_checkpoints = []
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.history_path = getattr(args, "history_path", None) or f"{args.algo}_history.json"
        self.plot_dir = getattr(args, "plot_dir", None) or os.path.join("plots", args.algo)
        self.history_flush_gap_steps = getattr(args, "history_flush_gap_steps", 10)
        self.plot_refresh_gap_steps = getattr(args, "plot_refresh_gap_steps", 50)
        self._record_lock = threading.Lock()
        self._step_event_count = 0

        history_dir = os.path.dirname(self.history_path)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        if self.plot_dir:
            os.makedirs(self.plot_dir, exist_ok=True)

        self.training_start_time = None
        self._effective_client_gpu_assignments = {}
        self._effective_client_batch_sizes = {}
        self.resume_round_idx = None

        if self.save_dir:
            self._refresh_saved_checkpoints()

    def _parse_client_gpu_map(self):
        entries = getattr(self.args, "client_gpu_map", None) or []
        if not entries:
            return {}

        mapping = {}
        for entry in entries:
            if "=" not in entry:
                raise ValueError(
                    f"Invalid --client_gpu_map entry '{entry}'. Expected format ClientName=g0[,g1,...]."
                )
            client_name, gpu_text = entry.split("=", 1)
            client_name = client_name.strip()
            if not client_name:
                raise ValueError(f"Invalid --client_gpu_map entry '{entry}': empty client name.")

            gpu_ids = []
            for token in gpu_text.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    gpu_id = int(token)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid GPU id '{token}' in --client_gpu_map entry '{entry}'."
                    ) from exc
                if gpu_id < 0:
                    raise ValueError(
                        f"Invalid GPU id '{gpu_id}' in --client_gpu_map entry '{entry}': must be >= 0."
                    )
                if gpu_id not in gpu_ids:
                    gpu_ids.append(gpu_id)

            if not gpu_ids:
                raise ValueError(
                    f"Invalid --client_gpu_map entry '{entry}': no usable GPU ids were provided."
                )
            mapping[client_name] = gpu_ids

        unknown_clients = sorted(set(mapping.keys()) - set(self.client_names))
        for client_name in unknown_clients:
            logger.warning("Ignoring --client_gpu_map entry for unknown client '%s'.", client_name)
            mapping.pop(client_name, None)
        return mapping

    def _parse_client_batch_size_map(self):
        entries = getattr(self.args, "client_batch_size_map", None) or []
        if not entries:
            return {}

        mapping = {}
        for entry in entries:
            if "=" not in entry:
                raise ValueError(
                    f"Invalid --client_batch_size_map entry '{entry}'. Expected format ClientName=batch_size."
                )
            client_name, batch_size_text = entry.split("=", 1)
            client_name = client_name.strip()
            if not client_name:
                raise ValueError(f"Invalid --client_batch_size_map entry '{entry}': empty client name.")

            try:
                batch_size = int(batch_size_text.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid batch size '{batch_size_text}' in --client_batch_size_map entry '{entry}'."
                ) from exc
            if batch_size <= 0:
                raise ValueError(
                    f"Invalid batch size '{batch_size}' in --client_batch_size_map entry '{entry}': must be > 0."
                )
            mapping[client_name] = batch_size

        unknown_clients = sorted(set(mapping.keys()) - set(self.client_names))
        for client_name in unknown_clients:
            logger.warning("Ignoring --client_batch_size_map entry for unknown client '%s'.", client_name)
            mapping.pop(client_name, None)
        return mapping

    def _filter_visible_gpu_ids(self, requested_gpu_ids):
        visible_count = torch.cuda.device_count()
        if visible_count <= 0:
            return []
        filtered = []
        for gpu_id in requested_gpu_ids:
            idx = int(gpu_id)
            if 0 <= idx < visible_count:
                if idx not in filtered:
                    filtered.append(idx)
            else:
                logger.warning(
                    "GPU %d is not visible (visible count=%d); ignoring this assignment.",
                    idx,
                    visible_count,
                )
        return filtered

    def get_server_state(self):
        return {}

    def load_server_state(self, state):
        pass

    def _ensure_history_keys(self):
        self.history.setdefault("clients", {})
        self.history.setdefault("val_accuracy", [])
        self.history.setdefault("val_macro_f1", [])
        self.history.setdefault("val_loss", [])
        self.history.setdefault("test_accuracy", [])
        self.history.setdefault("test_macro_f1", [])
        self.history.setdefault("test_loss", [])
        self.history.setdefault("round_elapsed_time", [])
        self.history.setdefault("eval_round", [])
        self.history.setdefault("eval_elapsed_time", [])
        self.history.setdefault("best_val_accuracy", None)
        self.history.setdefault("best_round", None)
        self.history.setdefault("early_stopped", False)
        self.history.setdefault("early_stopped_round", None)
        self._ensure_client_history()

    def _start_training_timer(self):
        self.training_start_time = time.time()

    def _elapsed_since_training_start(self):
        if self.training_start_time is None:
            return 0.0
        return float(time.time() - self.training_start_time)

    def _checkpoint_sort_key(self, path):
        name = os.path.basename(path)
        stem, _ = os.path.splitext(name)
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    def _refresh_saved_checkpoints(self):
        pattern = os.path.join(self.save_dir, "checkpoint_round_*.pth")
        self.saved_checkpoints = sorted(glob.glob(pattern), key=self._checkpoint_sort_key)

    def get_resume_start_round(self):
        if self.resume_round_idx is None:
            return 0
        return int(self.resume_round_idx) + 1

    def get_last_completed_round(self):
        start_round = self.get_resume_start_round()
        return start_round - 1 if start_round > 0 else -1


    def _ensure_client_history(self):
        self.history.setdefault("clients", {})
        for client in self.clients:
            client_history = self.history["clients"].setdefault(client.client_name, {})
            client_history.setdefault("train_step", [])
            client_history.setdefault("train_round", [])
            client_history.setdefault("train_epoch", [])
            client_history.setdefault("train_elapsed_time", [])
            client_history.setdefault("train_loss", [])
            client_history.setdefault("train_metrics", {})
            client_history.setdefault("val_round", [])
            client_history.setdefault("val_accuracy", [])
            client_history.setdefault("val_macro_f1", [])
            client_history.setdefault("val_loss", [])
            client_history.setdefault("test_round", [])
            client_history.setdefault("test_accuracy", [])
            client_history.setdefault("test_macro_f1", [])
            client_history.setdefault("test_loss", [])

    def record_client_train_step(self, client_name, round_idx, local_epoch, loss, metrics):
        with self._record_lock:
            self._ensure_client_history()
            client_history = self.history["clients"][client_name]
            next_step = len(client_history["train_step"])
            client_history["train_step"].append(next_step)
            client_history["train_round"].append(round_idx)
            client_history["train_epoch"].append(local_epoch)
            client_history["train_elapsed_time"].append(self._elapsed_since_training_start())
            client_history["train_loss"].append(float(loss))
            train_metrics = client_history.setdefault("train_metrics", {})
            for key, value in metrics.items():
                train_metrics.setdefault(key, []).append(float(value))
            self._step_event_count += 1
            self._maybe_flush_step_artifacts()

    def _append_client_eval_metrics(self, split, per_client_results, round_idx=None):
        if split not in {"val", "test"}:
            return
        self._ensure_client_history()
        for item in per_client_results:
            client_history = self.history["clients"][item["client_name"]]
            if round_idx is not None:
                client_history[f"{split}_round"].append(int(round_idx))
            client_history[f"{split}_accuracy"].append(item["accuracy"])
            client_history[f"{split}_macro_f1"].append(item["macro_f1"])
            client_history[f"{split}_loss"].append(item["loss"])

    def _history_for_json(self):
        compact_history = copy.deepcopy(self.history)
        compact_history.pop("eval_round", None)
        compact_history.pop("eval_elapsed_time", None)

        clients = compact_history.get("clients", {})
        for client_history in clients.values():
            client_history.pop("train_step", None)
            client_history.pop("train_round", None)
            client_history.pop("train_epoch", None)
            client_history.pop("train_elapsed_time", None)
            client_history.pop("train_loss", None)
            client_history.pop("train_metrics", None)

        return compact_history

    def _save_history_json(self):
        atomic_write_json(self._history_for_json(), self.history_path)

    def _save_history_plots(self):
        if self.plot_dir:
            save_training_curves(self.history, self.plot_dir, self.args.algo)

    def _maybe_flush_step_artifacts(self):
        if self.history_flush_gap_steps > 0 and self._step_event_count % self.history_flush_gap_steps == 0:
            self._save_history_json()
        if self.plot_refresh_gap_steps > 0 and self._step_event_count % self.plot_refresh_gap_steps == 0:
            self._save_history_plots()

    def flush_artifacts(self, force=False):
        del force
        self._save_history_json()
        self._save_history_plots()

    def _record_round_completion(self, round_idx, round_summary=None):
        elapsed_time = self._elapsed_since_training_start()
        self.history["round_elapsed_time"].append(elapsed_time)
        if round_summary is not None:
            self._record_evaluation(round_idx, round_summary, elapsed_time)
        return elapsed_time

    def _record_evaluation(self, round_idx, round_summary, elapsed_time=None):
        if elapsed_time is None:
            elapsed_time = self._elapsed_since_training_start()
        round_summary["elapsed_time"] = elapsed_time
        self.history["eval_round"].append(round_idx)
        self.history["eval_elapsed_time"].append(elapsed_time)
        return elapsed_time

    def _sync_early_stopping_history(self):
        self.history["best_val_accuracy"] = self.best_val_accuracy
        self.history["best_round"] = self.best_round
        self.history["early_stopped"] = self.early_stopped
        self.history["early_stopped_round"] = self.early_stopped_round

    def get_base_server_state(self):
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_round": self.best_round,
            "no_improvement_rounds": self.no_improvement_rounds,
            "early_stopped": self.early_stopped,
            "early_stopped_round": self.early_stopped_round,
            "best_state_snapshot": _clone_to_cpu(self.best_state_snapshot),
        }

    def load_base_server_state(self, state):
        state = state or {}
        self.best_val_accuracy = state.get("best_val_accuracy")
        self.best_round = state.get("best_round")
        self.no_improvement_rounds = state.get("no_improvement_rounds", 0)
        self.early_stopped = state.get("early_stopped", False)
        self.early_stopped_round = state.get("early_stopped_round")
        self.best_state_snapshot = state.get("best_state_snapshot")
        self._sync_early_stopping_history()

    def _snapshot_training_state(self):
        return {
            "server": _clone_to_cpu(self.get_server_state()),
            "clients": {client.client_name: client.get_state() for client in self.clients},
        }

    def _restore_training_state(self, snapshot):
        if not snapshot:
            return
        self.load_server_state(snapshot.get("server", {}))
        client_states = snapshot.get("clients", {})
        for client in self.clients:
            state = client_states.get(client.client_name)
            if state is not None:
                client.load_state(state)

    def _update_server_early_stopping(self, round_idx, val_summary):
        if not self.server_early_stopping_enabled:
            self._sync_early_stopping_history()
            return False

        val_accuracy = val_summary["accuracy"]
        improved = self.best_val_accuracy is None or val_accuracy > (self.best_val_accuracy + self.server_early_stopping_min_delta)
        if improved:
            self.best_val_accuracy = val_accuracy
            self.best_round = round_idx
            self.no_improvement_rounds = 0
            self.best_state_snapshot = self._snapshot_training_state()
            logger.info(
                "Validation accuracy improved to %.6f at round %d; updating federated early-stopping best state.",
                val_accuracy,
                round_idx,
            )
            self.save_best_checkpoint(round_idx)
            self._sync_early_stopping_history()
            return False

        self.no_improvement_rounds += 1
        if self.no_improvement_rounds >= self.server_early_stopping_patience:
            self.early_stopped = True
            self.early_stopped_round = round_idx
            logger.info(
                "Federated early stopping triggered at round %d; best val accuracy %.6f was reached at round %d.",
                round_idx,
                self.best_val_accuracy if self.best_val_accuracy is not None else float("nan"),
                self.best_round if self.best_round is not None else -1,
            )
        self._sync_early_stopping_history()
        return self.early_stopped

    def should_stop_training(self):
        return self.server_early_stopping_enabled and self.early_stopped

    def restore_best_state_for_final_test(self, fallback_round_idx):
        if not self.server_early_stopping_enabled or self.best_state_snapshot is None:
            return fallback_round_idx

        if self.best_round is not None and self.best_round != fallback_round_idx:
            logger.info(
                "Restoring best federated state from round %d for final test evaluation.",
                self.best_round,
            )
        self._restore_training_state(self.best_state_snapshot)
        return self.best_round if self.best_round is not None else fallback_round_idx

    def maybe_evaluate_validation(self, round_idx, elapsed_time=None):
        if self.eval_gap <= 0 or round_idx % self.eval_gap != 0:
            return None

        val_summary = self.evaluate(split="val", round_idx=round_idx)
        val_summary["round"] = round_idx
        self._record_evaluation(round_idx, val_summary, elapsed_time)
        self._print_round_summary(val_summary)
        self._update_server_early_stopping(round_idx, val_summary)
        self.flush_artifacts(force=True)
        return val_summary

    def set_clients(self, client_cls=Client):
        parallel = getattr(self.args, "parallel", False)
        requested_gpu_map = self._parse_client_gpu_map()
        requested_batch_map = self._parse_client_batch_size_map()

        if parallel:
            num_gpus = torch.cuda.device_count()
            if num_gpus < 2:
                logger.warning(
                    "parallel=True but only %d GPU(s) detected. "
                    "Falling back to serial mode.",
                    num_gpus,
                )
                self.args.parallel = False
                client_devices = [self.args.device] * len(self.client_names)
            else:
                client_devices = [
                    torch.device(f"cuda:{i % num_gpus}")
                    for i in range(len(self.client_names))
                ]
                logger.info(
                    "parallel mode: %d clients assigned to %d GPU(s): %s",
                    len(self.client_names),
                    num_gpus,
                    [str(d) for d in client_devices],
                )
        else:
            client_devices = [self.args.device] * len(self.client_names)

        client_devices = [torch.device(device) for device in client_devices]
        effective_gpu_assignments = {}
        for index, client_name in enumerate(self.client_names):
            requested_gpu_ids = requested_gpu_map.get(client_name)
            if requested_gpu_ids is not None:
                filtered_gpu_ids = self._filter_visible_gpu_ids(requested_gpu_ids)
                if filtered_gpu_ids:
                    client_devices[index] = torch.device(f"cuda:{filtered_gpu_ids[0]}")
                    effective_gpu_assignments[client_name] = tuple(filtered_gpu_ids)
                else:
                    fallback_device = client_devices[index]
                    if fallback_device.type == "cuda":
                        fallback_idx = 0 if fallback_device.index is None else int(fallback_device.index)
                        effective_gpu_assignments[client_name] = (fallback_idx,)
                    else:
                        effective_gpu_assignments[client_name] = ()
            else:
                assigned_device = client_devices[index]
                if assigned_device.type == "cuda":
                    fallback_idx = 0 if assigned_device.index is None else int(assigned_device.index)
                    effective_gpu_assignments[client_name] = (fallback_idx,)
                else:
                    effective_gpu_assignments[client_name] = ()

        self.clients = [
            client_cls(self.args, client_name=name, device=device)
            for name, device in zip(self.client_names, client_devices)
        ]

        effective_batch_sizes = {}
        for client in self.clients:
            mapped_batch_size = requested_batch_map.get(client.client_name)
            if mapped_batch_size is not None:
                client.batch_size = int(mapped_batch_size)
            effective_batch_sizes[client.client_name] = int(client.batch_size)

            assigned_gpu_ids = effective_gpu_assignments.get(client.client_name, ())
            if assigned_gpu_ids:
                client.brats_ddp_devices = list(assigned_gpu_ids)
                if len(assigned_gpu_ids) > 1:
                    client.brats_ddp = True
                    if client.client_name != "BraTS":
                        client.ddp_force_enabled = True
                elif client.client_name != "BraTS":
                    client.ddp_force_enabled = False

        for client in self.clients:
            client.set_train_step_recorder(self.record_client_train_step)

        self._effective_client_gpu_assignments = effective_gpu_assignments
        self._effective_client_batch_sizes = effective_batch_sizes
        if requested_gpu_map:
            pretty_gpu_map = {
                name: (",".join(str(idx) for idx in gpu_ids) if gpu_ids else "cpu")
                for name, gpu_ids in effective_gpu_assignments.items()
            }
            logger.info("Effective client GPU assignments: %s", pretty_gpu_map)
        if requested_batch_map:
            logger.info("Effective client batch sizes: %s", effective_batch_sizes)
        self._ensure_client_history()

    def select_clients(self):
        if self.include_yale:
            self.selected_clients = list(self.clients)
        else:
            self.selected_clients = [
                client for client in self.clients if client.client_name != "Yale"
            ]
        return self.selected_clients

    def get_client_payload(self):
        return None

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.clients:
            client.set_parameters(payload)

    def receive_ids(self):
        if not self.selected_clients:
            raise RuntimeError("No selected clients to receive from.")

        active_clients = list(self.selected_clients)
        total_samples = sum(client.train_samples for client in active_clients)
        self.uploaded_client_names = [client.client_name for client in active_clients]
        self.uploaded_weights = [
            client.train_samples / max(total_samples, 1) for client in active_clients
        ]
        return active_clients

    def aggregate(self, active_clients):
        raise NotImplementedError

    def train_clients(self, clients, round_idx):
        use_parallel_training = bool(getattr(self.args, "parallel", False)) or bool(getattr(self.args, "client_gpu_map", None))
        if use_parallel_training:
            self._train_clients_parallel(clients, round_idx)
        else:
            self._train_clients_serial(clients, round_idx)

    def _train_clients_serial(self, clients, round_idx):
        for i, client in enumerate(clients):
            client.activate()
            try:
                client.train(round_idx, i)
            finally:
                client.offload()

    def _train_clients_parallel(self, clients, round_idx):
        def _run_client_train(client, client_idx):
            assigned_gpu_ids = self._effective_client_gpu_assignments.get(client.client_name, ())
            gpu_text = ",".join(str(gid) for gid in assigned_gpu_ids) if assigned_gpu_ids else "cpu"
            logger.info(
                "Round %d train start: %-12s (gpu=%s, batch_size=%d)",
                round_idx,
                client.client_name,
                gpu_text,
                int(client.batch_size),
            )
            client.activate()
            try:
                client.train(round_idx, client_idx)
                logger.info("Round %d train done : %s", round_idx, client.client_name)
            finally:
                client.offload()

        ddp_clients = []
        threaded_clients = []
        for i, client in enumerate(clients):
            use_main_thread = False
            should_use_ddp_fn = getattr(client, "_should_use_brats_ddp", None)
            if callable(should_use_ddp_fn):
                try:
                    use_main_thread = bool(should_use_ddp_fn())
                except Exception:
                    use_main_thread = False
            if use_main_thread:
                ddp_clients.append((i, client))
            else:
                threaded_clients.append((i, client))

        if ddp_clients:
            logger.info(
                "Round %d: running %d DDP client(s) in main thread to avoid spawn-in-thread deadlocks.",
                round_idx,
                len(ddp_clients),
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(threaded_clients), 1)) as executor:
            futures = {
                executor.submit(_run_client_train, client, i): client
                for i, client in threaded_clients
            }

            main_thread_error = None
            for i, client in ddp_clients:
                try:
                    _run_client_train(client, i)
                except Exception as exc:
                    main_thread_error = RuntimeError(
                        f"Client {client.client_name} raised an exception during training"
                    )
                    main_thread_error.__cause__ = exc
                    break

            future_error = None
            for future in concurrent.futures.as_completed(futures):
                client = futures[future]
                exc = future.exception()
                if exc is not None:
                    future_error = RuntimeError(
                        f"Client {client.client_name} raised an exception during training"
                    )
                    future_error.__cause__ = exc
                    break

            if main_thread_error is not None:
                raise main_thread_error
            if future_error is not None:
                raise future_error
        print()

    def _print_round_summary(self, round_summary):
        logger.info("─" * 40)
        logger.info("Round %d Evaluation (%s):", round_summary["round"], round_summary["split"])
        logger.info(
            "  Avg Acc: %.4f | Avg F1: %.4f | Loss: %.4f",
            round_summary["accuracy"], round_summary["macro_f1"], round_summary["loss"]
        )
        for item in round_summary["per_client_results"]:
            logger.info(
                "    - %-12s: Acc=%.4f, F1=%.4f (n=%d)",
                item["client_name"], item["accuracy"], item["macro_f1"], item["num_samples"]
            )
        logger.info("─" * 40)

    def _build_checkpoint_payload(self, round_idx):
        return {
            "round_idx": round_idx,
            "history": self.history,
            "clients": {client.client_name: client.get_state() for client in self.clients},
            "server": self.get_server_state(),
            "server_base": self.get_base_server_state(),
        }

    def save_checkpoint(self, round_idx):
        if not self.save_dir:
            return

        checkpoint = self._build_checkpoint_payload(round_idx)
        save_path = os.path.join(self.save_dir, f"checkpoint_round_{round_idx}.pth")
        torch.save(checkpoint, save_path)
        logger.info("Checkpoint saved to %s", save_path)

        self._refresh_saved_checkpoints()

        while len(self.saved_checkpoints) > self.save_total_limit:
            old_ckpt = self.saved_checkpoints.pop(0)
            if os.path.exists(old_ckpt):
                try:
                    os.remove(old_ckpt)
                    logger.info("Removed old checkpoint: %s", old_ckpt)
                except Exception as e:
                    logger.warning("Failed to remove old checkpoint %s: %s", old_ckpt, e)

    def save_best_checkpoint(self, round_idx):
        if not self.save_dir:
            return
        checkpoint = self._build_checkpoint_payload(round_idx)
        save_path = os.path.join(self.save_dir, "best_checkpoint.pth")
        torch.save(checkpoint, save_path)
        logger.info("Best checkpoint saved to %s", save_path)

    def maybe_save_checkpoint(self, round_idx):
        if self.save_gap > 0 and round_idx % self.save_gap == 0:
            self.save_checkpoint(round_idx)

    def load_checkpoint(self, load_path):
        if not load_path or not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")
        self.history = checkpoint.get("history", self.history)
        self._ensure_history_keys()
        self.load_server_state(checkpoint.get("server", {}))
        self.load_base_server_state(checkpoint.get("server_base", {}))

        client_states = checkpoint.get("clients", {})
        for client in self.clients:
            if client.client_name in client_states:
                client.load_state(client_states[client.client_name])

        round_idx = checkpoint.get("round_idx", 0)
        self.resume_round_idx = int(round_idx)
        logger.info("Successfully loaded checkpoint from %s (Round %d)", load_path, round_idx)
        logger.info("Resuming training from round %d", self.resume_round_idx + 1)
        self._refresh_saved_checkpoints()
        return round_idx

    def train_round(self, round_idx):
        raise NotImplementedError

    def evaluate(self, split="test", round_idx=None):
        target_clients = self.selected_clients if self.selected_clients else self.select_clients()
        metrics = []
        round_text = "final" if round_idx is None else str(round_idx)
        for client in target_clients:
            logger.info("Round %s eval(%s) start: %s", round_text, split, client.client_name)
            if getattr(self.args, "parallel", False):
                item = client.evaluate_split(split)
            else:
                client.activate()
                try:
                    item = client.evaluate_split(split)
                finally:
                    client.offload()
            metrics.append(item)
            logger.info("Round %s eval(%s) done : %s", round_text, split, client.client_name)
        total_samples = sum(item["num_samples"] for item in metrics)

        avg_accuracy = sum(
            item["accuracy"] * item["num_samples"] for item in metrics
        ) / max(total_samples, 1)
        avg_macro_f1 = sum(
            item["macro_f1"] * item["num_samples"] for item in metrics
        ) / max(total_samples, 1)
        avg_loss = sum(
            item["loss"] * item["num_samples"] for item in metrics
        ) / max(total_samples, 1)

        per_client_results = []
        for client, item in zip(target_clients, metrics):
            per_client_results.append(
                {
                    "client_name": client.client_name,
                    "num_samples": item["num_samples"],
                    "loss": item["loss"],
                    "accuracy": item["accuracy"],
                    "macro_f1": item["macro_f1"],
                }
            )

        per_client_acc = [item["accuracy"] for item in per_client_results]
        per_client_f1 = [item["macro_f1"] for item in per_client_results]
        client_names = [item["client_name"] for item in per_client_results]

        summary = {
            "split": split,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "macro_f1": avg_macro_f1,
            "std_accuracy": float(np.std(per_client_acc)) if per_client_acc else 0.0,
            "std_macro_f1": float(np.std(per_client_f1)) if per_client_f1 else 0.0,
            "client_names": client_names,
            "per_client_results": per_client_results,
        }

        self._append_client_eval_metrics(split, per_client_results, round_idx=round_idx)
        if split == "val":
            self.history["val_accuracy"].append(avg_accuracy)
            self.history["val_macro_f1"].append(avg_macro_f1)
            self.history["val_loss"].append(avg_loss)
        elif split == "test":
            self.history["test_accuracy"].append(avg_accuracy)
            self.history["test_macro_f1"].append(avg_macro_f1)
            self.history["test_loss"].append(avg_loss)
        return summary
