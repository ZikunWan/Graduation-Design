import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from client.clientbase import _clone_to_cpu
from client.clientlocal import ClientLocal
from .serverbase import Server

logger = logging.getLogger(__name__)


class ServerLocal(Server):
    def __init__(self, args):
        super().__init__(args)
        if getattr(args, "local_epochs", 1) != 1:
            logger.info(
                "algo=local ignores local_epochs=%d; using one local epoch per round so global_rounds equals total local epochs.",
                args.local_epochs,
            )
        self.set_clients(ClientLocal)
        self._client_batch_sizes = {}
        self._client_gpu_assignments = {}
        self._gpu_locks = {}
        self._checkpoint_lock = threading.Lock()
        self._client_saved_round_checkpoints = {}
        self._configure_client_batch_sizes()
        self._configure_client_gpu_assignments()
        self.history = self._build_local_history()
        self._client_saved_round_checkpoints = {client.client_name: [] for client in self.clients}

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

    def _configure_client_batch_sizes(self):
        requested_map = self._parse_client_batch_size_map()
        default_batch_size = int(self.args.batch_size)
        self._client_batch_sizes = {}
        for client in self.clients:
            batch_size = int(requested_map.get(client.client_name, default_batch_size))
            client.batch_size = batch_size
            self._client_batch_sizes[client.client_name] = batch_size
        logger.info("Local client batch sizes: %s", self._client_batch_sizes)

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

    def _resolve_default_gpu_ids(self):
        if not torch.cuda.is_available():
            return []
        default_device = torch.device(self.args.device)
        if default_device.type != "cuda":
            return []
        if default_device.index is None:
            return [0]
        return [int(default_device.index)]

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

    def _configure_client_gpu_assignments(self):
        requested_map = self._parse_client_gpu_map()
        default_gpu_ids = self._resolve_default_gpu_ids()
        default_gpu_ids = self._filter_visible_gpu_ids(default_gpu_ids)
        if torch.cuda.is_available() and not default_gpu_ids:
            default_gpu_ids = [0]

        self._client_gpu_assignments = {}
        for client in self.clients:
            requested_gpu_ids = requested_map.get(client.client_name, list(default_gpu_ids))
            gpu_ids = self._filter_visible_gpu_ids(requested_gpu_ids)

            if not gpu_ids and default_gpu_ids and requested_gpu_ids != default_gpu_ids:
                gpu_ids = list(default_gpu_ids)

            if gpu_ids:
                client.device = torch.device(f"cuda:{gpu_ids[0]}")
                client.pin_memory = True
                client.brats_ddp_devices = list(gpu_ids)
                if len(gpu_ids) > 1:
                    client.brats_ddp = True
                    client.ddp_force_enabled = True
                else:
                    client.ddp_force_enabled = False
            else:
                client.device = torch.device("cpu")
                client.pin_memory = False
                client.brats_ddp_devices = []
                client.ddp_force_enabled = False

            self._client_gpu_assignments[client.client_name] = tuple(gpu_ids)

        assigned_gpu_ids = sorted(
            {
                gpu_id
                for gpu_ids in self._client_gpu_assignments.values()
                for gpu_id in gpu_ids
            }
        )
        self._gpu_locks = {gpu_id: threading.Lock() for gpu_id in assigned_gpu_ids}

        assignment_desc = {
            client.client_name: (
                ",".join(str(gpu_id) for gpu_id in self._client_gpu_assignments[client.client_name])
                if self._client_gpu_assignments[client.client_name]
                else "cpu"
            )
            for client in self.clients
        }
        logger.info("Local client GPU assignments: %s", assignment_desc)

    def _build_local_history(self):
        return {
            "clients": {
                client.client_name: {
                    "train_step": [],
                    "train_round": [],
                    "train_epoch": [],
                    "train_elapsed_time": [],
                    "train_loss": [],
                    "train_metrics": {},
                    "val_round": [],
                    "val_accuracy": [],
                    "val_macro_f1": [],
                    "val_loss": [],
                    "test_round": [],
                    "best_val_accuracy": None,
                    "best_round": None,
                    "early_stopped": False,
                    "early_stopped_round": None,
                    "final_test_accuracy": None,
                    "final_test_macro_f1": None,
                    "final_test_loss": None,
                    "final_test_num_samples": None,
                }
                for client in self.clients
            },
        }

    def _ensure_local_history(self):
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
            client_history.setdefault("best_val_accuracy", None)
            client_history.setdefault("best_round", None)
            client_history.setdefault("early_stopped", False)
            client_history.setdefault("early_stopped_round", None)
            client_history.setdefault("final_test_accuracy", None)
            client_history.setdefault("final_test_macro_f1", None)
            client_history.setdefault("final_test_loss", None)
            client_history.setdefault("final_test_num_samples", None)

    def load_checkpoint(self, load_path):
        if not load_path or not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")
        self.history = checkpoint.get("history", self._build_local_history())
        self._ensure_local_history()
        self.load_server_state(checkpoint.get("server", {}))

        client_states = checkpoint.get("clients", {})
        for client in self.clients:
            if client.client_name in client_states:
                client.load_state(client_states[client.client_name])
            client_history = self.history["clients"].get(client.client_name, {})
            client.best_val_accuracy = client_history.get("best_val_accuracy")
            client.best_round = client_history.get("best_round")
            client.early_stopped = client_history.get("early_stopped", False)
            client.early_stopped_round = client_history.get("early_stopped_round")

        round_idx = checkpoint.get("round_idx", 0)
        logger.info("Successfully loaded checkpoint from %s (Round %d)", load_path, round_idx)
        return round_idx

    def aggregate(self, active_clients):
        del active_clients
        return None

    def evaluate(self, split="test", round_idx=None):
        del round_idx
        target_clients = self.selected_clients if self.selected_clients else self.select_clients()
        metrics = []
        for client in target_clients:
            if getattr(self.args, "parallel", False):
                item = client.evaluate_split(split)
            else:
                client.activate()
                try:
                    item = client.evaluate_split(split)
                finally:
                    client.offload()
            metrics.append(item)

        total_samples = sum(item["num_samples"] for item in metrics)
        avg_accuracy = sum(item["accuracy"] * item["num_samples"] for item in metrics) / max(total_samples, 1)
        avg_macro_f1 = sum(item["macro_f1"] * item["num_samples"] for item in metrics) / max(total_samples, 1)
        avg_loss = sum(item["loss"] * item["num_samples"] for item in metrics) / max(total_samples, 1)

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

        return {
            "split": split,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "macro_f1": avg_macro_f1,
            "std_accuracy": float(np.std(per_client_acc)) if per_client_acc else 0.0,
            "std_macro_f1": float(np.std(per_client_f1)) if per_client_f1 else 0.0,
            "client_names": [item["client_name"] for item in per_client_results],
            "per_client_results": per_client_results,
        }

    def _sync_client_status(self):
        for client in self.clients:
            client_history = self.history["clients"][client.client_name]
            client_history["best_val_accuracy"] = client.best_val_accuracy
            client_history["best_round"] = client.best_round
            client_history["early_stopped"] = client.early_stopped
            client_history["early_stopped_round"] = client.early_stopped_round

    def _record_validation_summary(self, round_idx=None):
        val_summary = self.evaluate(split="val")
        for item in val_summary["per_client_results"]:
            client_history = self.history["clients"][item["client_name"]]
            if round_idx is None:
                client_history["val_round"].append(len(client_history["val_round"]))
            else:
                client_history["val_round"].append(int(round_idx))
            client_history["val_accuracy"].append(item["accuracy"])
            client_history["val_macro_f1"].append(item["macro_f1"])
            client_history["val_loss"].append(item["loss"])
        self._sync_client_status()
        self.flush_artifacts(force=True)
        return val_summary

    def _record_single_client_validation(self, client, round_idx):
        metrics = client.last_val_metrics
        if metrics is None:
            client.activate()
            try:
                metrics = client.evaluate_split("val")
            finally:
                client.offload()

        with self._record_lock:
            client_history = self.history["clients"][client.client_name]
            client_history["val_round"].append(int(round_idx))
            client_history["val_accuracy"].append(float(metrics["accuracy"]))
            client_history["val_macro_f1"].append(float(metrics["macro_f1"]))
            client_history["val_loss"].append(float(metrics["loss"]))
            client_history["best_val_accuracy"] = client.best_val_accuracy
            client_history["best_round"] = client.best_round
            client_history["early_stopped"] = client.early_stopped
            client_history["early_stopped_round"] = client.early_stopped_round
            self.flush_artifacts(force=True)

    def _record_final_test_summary(self, test_summary):
        for item in test_summary["per_client_results"]:
            client_history = self.history["clients"][item["client_name"]]
            test_round = int(test_summary.get("round", len(client_history.get("test_round", []))))
            client_history.setdefault("test_round", []).append(test_round)
            client_history["final_test_accuracy"] = item["accuracy"]
            client_history["final_test_macro_f1"] = item["macro_f1"]
            client_history["final_test_loss"] = item["loss"]
            client_history["final_test_num_samples"] = item["num_samples"]
        self._sync_client_status()

    def save_checkpoint(self, round_idx):
        super().save_checkpoint(round_idx)
        if not self.save_dir:
            return

        for client in self.clients:
            client_history = self.history["clients"][client.client_name]
            final_path = os.path.join(self.save_dir, f"{client.client_name}_final.pth")
            final_checkpoint = {
                "checkpoint_type": "final",
                "client_name": client.client_name,
                "round_idx": round_idx,
                "history": client_history,
                "client_state": client.get_state(),
            }
            torch.save(final_checkpoint, final_path)
            logger.info("Local final checkpoint saved to %s", final_path)

            best_state = client.best_model_state if client.best_model_state is not None else client.get_model_state()
            best_optimizer = (
                _clone_to_cpu(client.best_optimizer_state)
                if getattr(client, "best_optimizer_state", None) is not None
                else _clone_to_cpu(client.optimizer.state_dict())
            )
            best_path = os.path.join(self.save_dir, f"{client.client_name}_best.pth")
            best_checkpoint = {
                "checkpoint_type": "best",
                "client_name": client.client_name,
                "best_round": client.best_round,
                "best_val_accuracy": client.best_val_accuracy,
                "history": client_history,
                "client_state": {
                    "model": _clone_to_cpu(best_state),
                    "optimizer": best_optimizer,
                },
            }
            torch.save(best_checkpoint, best_path)
            logger.info("Local best checkpoint saved to %s", best_path)

    def _save_client_round_checkpoint(self, client, round_idx):
        if not self.save_dir:
            return

        client_history = self.history["clients"][client.client_name]
        save_path = os.path.join(self.save_dir, f"{client.client_name}_checkpoint_round_{round_idx}.pth")
        checkpoint = {
            "checkpoint_type": "round",
            "client_name": client.client_name,
            "round_idx": int(round_idx),
            "history": client_history,
            "client_state": client.get_state(),
        }

        with self._checkpoint_lock:
            torch.save(checkpoint, save_path)
            logger.info("Local round checkpoint saved to %s", save_path)
            saved = self._client_saved_round_checkpoints.setdefault(client.client_name, [])
            saved.append(save_path)
            while len(saved) > self.save_total_limit:
                old_path = saved.pop(0)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        logger.info("Removed old local round checkpoint: %s", old_path)
                    except Exception as exc:
                        logger.warning("Failed to remove old local round checkpoint %s: %s", old_path, exc)

    def train_round(self, round_idx):
        if not self.selected_clients:
            raise RuntimeError("ServerLocal.train_round requires exactly one selected client.")
        self.train_clients(self.selected_clients, round_idx)
        self._record_validation_summary(round_idx)
        self.maybe_save_checkpoint(round_idx)

    def _train_single_client(self, client, client_idx):
        last_round_idx = -1
        assigned_gpu_ids = self._client_gpu_assignments.get(client.client_name, ())
        assignment_label = ",".join(str(gpu_id) for gpu_id in assigned_gpu_ids) if assigned_gpu_ids else "cpu"
        logger.info("Training local client %s on %s", client.client_name, assignment_label)
        try:
            for round_idx in range(self.global_rounds):
                last_round_idx = round_idx
                client.activate()
                try:
                    client.train(round_idx, client_idx)
                finally:
                    client.offload()
                self._record_single_client_validation(client, round_idx)
                if self.save_gap > 0 and round_idx % self.save_gap == 0:
                    self._save_client_round_checkpoint(client, round_idx)
                if client.early_stopped:
                    logger.info(
                        "Local client %s finished early at round %d.",
                        client.client_name,
                        round_idx,
                    )
                    break
        finally:
            if hasattr(client, "shutdown_ddp_workers"):
                client.shutdown_ddp_workers()
        return last_round_idx

    def _train_single_client_with_gpu_lock(self, client, client_idx):
        gpu_ids = sorted(set(self._client_gpu_assignments.get(client.client_name, ())))
        held_locks = []
        try:
            for gpu_id in gpu_ids:
                lock = self._gpu_locks[gpu_id]
                lock.acquire()
                held_locks.append(lock)
            return self._train_single_client(client, client_idx)
        finally:
            for lock in reversed(held_locks):
                lock.release()

    def train(self):
        self._start_training_timer()
        target_clients = self.select_clients()
        last_round_idx = -1

        with ThreadPoolExecutor(max_workers=max(len(target_clients), 1)) as executor:
            futures = {
                executor.submit(self._train_single_client_with_gpu_lock, client, idx): client
                for idx, client in enumerate(target_clients)
            }
            for future in as_completed(futures):
                client = futures[future]
                try:
                    client_last_round_idx = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Local client {client.client_name} raised an exception during training."
                    ) from exc
                last_round_idx = max(last_round_idx, client_last_round_idx)

        self.selected_clients = target_clients
        final_summary = self.evaluate(split="test", round_idx=last_round_idx if last_round_idx >= 0 else 0)
        final_summary["round"] = last_round_idx
        self._record_final_test_summary(final_summary)
        self._print_round_summary(final_summary)
        if last_round_idx >= 0:
            self.save_checkpoint(last_round_idx)
        return [final_summary]
