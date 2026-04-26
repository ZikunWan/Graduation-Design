import copy
import io
import logging
import os
import socket
import tempfile
import threading
import warnings
from collections import Counter
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset import BrainTumorCaseDataset
from model import build_client_model

logger = logging.getLogger(__name__)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_ddp_dataloader(client, dataset, sampler, drop_last=False):
    # In DDP spawn workers, nested DataLoader multiprocessing can deadlock on
    # some Python/PyTorch/runtime combinations. Prefer single-process loading
    # inside each DDP rank for stability.
    effective_num_workers = client.num_workers
    if mp.current_process().name != "MainProcess":
        effective_num_workers = 0

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": client.batch_size,
        "shuffle": False,
        "drop_last": drop_last,
        "sampler": sampler,
        "collate_fn": client._get_collate_fn(dataset),
        "num_workers": effective_num_workers,
        "pin_memory": client.pin_memory,
    }
    if effective_num_workers > 0:
        dataloader_kwargs["persistent_workers"] = client.persistent_workers
        dataloader_kwargs["prefetch_factor"] = client.prefetch_factor
    return DataLoader(**dataloader_kwargs)


def _configure_ddp_worker_output(rank):
    if rank == 0:
        return
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def _serialize_queue_payload(payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _deserialize_queue_payload(payload_bytes):
    return torch.load(io.BytesIO(payload_bytes), map_location="cpu")


def _create_ddp_result_path(prefix):
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".pt")
    os.close(fd)
    return path


def _load_and_cleanup_ddp_result(path):
    try:
        return torch.load(path, map_location="cpu")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _clone_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _move_optimizer_value(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_optimizer_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_optimizer_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_optimizer_value(item, device) for item in value)
    return value


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_optimizer_value(value, device)


def _client_eval_ddp_worker(rank, world_size, worker_config, result_queue):
    _configure_ddp_worker_output(rank)
    device_ids = worker_config["device_ids"]
    device_id = device_ids[rank]
    split = worker_config["split"]

    args = copy.deepcopy(worker_config["args"])
    args.device = f"cuda:{device_id}"
    args.parallel = False
    args.brats_ddp = False

    seed = int(worker_config["seed"]) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device_id)

    dist.init_process_group(
        backend=worker_config["backend"],
        init_method=worker_config["init_method"],
        world_size=world_size,
        rank=rank,
    )

    try:
        client = Client(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_model_state(worker_config["model_state"])
        client.model.to(client.device)
        client.is_model_materialized = True
        client.enable_model_offload = False
        client.model.eval()

        dataset = client.datasets[split]
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = _build_ddp_dataloader(client, dataset, sampler, drop_last=False)

        total_loss = torch.tensor(0.0, device=client.device, dtype=torch.float64)
        total_correct = torch.tensor(0.0, device=client.device, dtype=torch.float64)
        total_samples = torch.tensor(0.0, device=client.device, dtype=torch.float64)
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y in dataloader:
                x = client._move_to_device(x)
                y = y.to(client.device, non_blocking=client.pin_memory)
                with client.autocast_context():
                    logits = client.model(x)
                    loss = client.loss_fn(logits, y)

                preds = torch.argmax(logits, dim=1)
                batch_size = y.shape[0]
                total_loss += loss.detach().to(dtype=torch.float64) * float(batch_size)
                total_correct += (preds == y).sum().detach().to(dtype=torch.float64)
                total_samples += float(batch_size)

                all_labels.extend(y.detach().cpu().tolist())
                all_preds.extend(preds.detach().cpu().tolist())

        packed_metrics = torch.stack([total_loss, total_correct, total_samples], dim=0)
        dist.all_reduce(packed_metrics, op=dist.ReduceOp.SUM)

        gathered_labels = [None for _ in range(world_size)]
        gathered_preds = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_labels, all_labels)
        dist.all_gather_object(gathered_preds, all_preds)

        if rank == 0:
            merged_labels = []
            merged_preds = []
            for labels in gathered_labels:
                if labels:
                    merged_labels.extend(labels)
            for preds in gathered_preds:
                if preds:
                    merged_preds.extend(preds)

            total_samples_value = int(packed_metrics[2].item())
            avg_loss = float(packed_metrics[0].item() / max(total_samples_value, 1))
            accuracy = float(packed_metrics[1].item() / max(total_samples_value, 1))
            macro_f1 = (
                float(f1_score(merged_labels, merged_preds, average="macro", zero_division=0))
                if total_samples_value > 0
                else 0.0
            )
            result_queue.put(
                _serialize_queue_payload(
                    {
                        "loss": avg_loss,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "num_samples": total_samples_value,
                    }
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class Client:
    def __init__(self, args, client_name, device=None, **kwargs):
        self.args = args
        self.client_name = client_name
        self.device = torch.device(device if device is not None else args.device)
        self.root_dir = args.root_dir
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.model_name = args.model_name
        self.prototype_dim = args.prototype_dim
        self.dropout = args.dropout
        self.val_ratio = args.val_ratio
        self.seed = args.seed
        self.parallel = getattr(args, "parallel", False)
        self.model_mode = getattr(args, "model_mode", "auto")
        self.enable_model_offload = not self.parallel and self.device.type == "cuda"
        self.amp_enabled = bool(getattr(args, "amp", True)) and self.device.type == "cuda"
        self.num_workers = max(int(getattr(args, "num_workers", 4)), 0)
        self.pin_memory = bool(getattr(args, "pin_memory", True)) and self.device.type == "cuda"
        self.persistent_workers = bool(getattr(args, "persistent_workers", True)) and self.num_workers > 0
        self.prefetch_factor = max(int(getattr(args, "prefetch_factor", 2)), 1)
        self.brats_ddp = bool(getattr(args, "brats_ddp", False))
        self.brats_ddp_backend = getattr(args, "brats_ddp_backend", "nccl")
        self.brats_ddp_devices = getattr(args, "brats_ddp_devices", None)
        self.ddp_force_enabled = False
        self._ddp_warning_emitted = False
        self._threaded_dataloader_warning_emitted = False

        max_samples = getattr(args, "max_samples", None)
        full_train_dataset = BrainTumorCaseDataset(
            split="train",
            client_name=self.client_name,
            root_dir=self.root_dir,
            max_samples=max_samples,
        )
        self.train_dataset, self.val_dataset = self._split_train_val_dataset(full_train_dataset)
        self.test_dataset = BrainTumorCaseDataset(
            split="test",
            client_name=self.client_name,
            root_dir=self.root_dir,
            max_samples=max_samples,
        )

        self.datasets = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test": self.test_dataset,
        }

        self.train_samples = len(self.train_dataset)
        self.val_samples = len(self.val_dataset)
        self.test_samples = len(self.test_dataset)

        initial_device = torch.device("cpu") if self.enable_model_offload else self.device
        self.model = build_client_model(
            client_name=self.client_name,
            num_classes=self.num_classes,
            model_name=self.model_name,
            prototype_dim=self.prototype_dim,
            dropout=self.dropout,
            model_mode=self.model_mode,
            algo=getattr(args, "algo", None),
        ).to(initial_device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.local_learning_rate,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.is_model_materialized = not self.enable_model_offload
        self.train_step_recorder = None

    def activate(self):
        if not self.enable_model_offload or self.is_model_materialized:
            return
        self.model.to(self.device)
        move_optimizer_state(self.optimizer, self.device)
        self.is_model_materialized = True

    def offload(self):
        if not self.enable_model_offload or not self.is_model_materialized:
            return
        cpu_device = torch.device("cpu")
        self.model.to(cpu_device)
        move_optimizer_state(self.optimizer, cpu_device)
        self.is_model_materialized = False
        torch.cuda.empty_cache()

    def _split_train_val_dataset(self, dataset):
        num_samples = len(dataset)
        if self.val_ratio <= 0 or num_samples == 0:
            return dataset, Subset(dataset, [])

        if num_samples == 1:
            return Subset(dataset, [0]), Subset(dataset, [])

        indices = list(range(num_samples))
        labels = [sample["label"] for sample in dataset.samples]
        unique_labels = set(labels)
        label_counts = Counter(labels)
        use_stratify = len(unique_labels) > 1 and min(label_counts.values()) >= 2
        stratify_labels = labels if use_stratify else None

        train_indices, val_indices = train_test_split(
            indices,
            test_size=self.val_ratio,
            random_state=self.seed,
            shuffle=True,
            stratify=stratify_labels,
        )

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        return train_dataset, val_dataset

    def _get_collate_fn(self, dataset):
        if isinstance(dataset, Subset):
            return dataset.dataset.get_collate_fn()
        return dataset.get_collate_fn()

    def _resolve_dataloader_num_workers(self):
        if self.num_workers <= 0:
            return 0
        if threading.current_thread() is not threading.main_thread():
            if not self._threaded_dataloader_warning_emitted:
                logger.warning(
                    "%s DataLoader is created in a worker thread; forcing num_workers=0 to avoid multiprocessing deadlocks.",
                    self.client_name,
                )
                self._threaded_dataloader_warning_emitted = True
            return 0
        return self.num_workers

    def load_data(self, split, batch_size=None, shuffle=False, drop_last=False):
        dataset = self.datasets[split]
        current_batch_size = self.batch_size if batch_size is None else batch_size
        effective_num_workers = self._resolve_dataloader_num_workers()
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": current_batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "collate_fn": self._get_collate_fn(dataset),
            "num_workers": effective_num_workers,
            "pin_memory": self.pin_memory,
        }
        if effective_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**dataloader_kwargs)

    def load_train_data(self, batch_size=None, shuffle=True, drop_last=False):
        return self.load_data(
            split="train",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def load_val_data(self, batch_size=None, shuffle=False, drop_last=False):
        return self.load_data(
            split="val",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def load_test_data(self, batch_size=None, shuffle=False):
        return self.load_data(
            split="test",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _move_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=self.pin_memory)
        if isinstance(batch, dict):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_to_device(value) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_to_device(value) for value in batch)
        return batch

    def autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def backward_step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def get_model_state(self):
        return _clone_to_cpu(self.model.state_dict())

    def load_model_state(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def get_state(self):
        return {
            "model": _clone_to_cpu(self.model.state_dict()),
            "optimizer": _clone_to_cpu(self.optimizer.state_dict()),
            "scaler": _clone_to_cpu(self.grad_scaler.state_dict()),
        }

    def load_state(self, state):
        self.model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            if self.is_model_materialized:
                move_optimizer_state(self.optimizer, self.device)
        scaler_state = state.get("scaler")
        if scaler_state:
            self.grad_scaler.load_state_dict(scaler_state)

    def set_parameters(self, payload):
        if payload is None:
            return
        state_dict = payload.get("model_state")
        if state_dict is not None:
            self.load_model_state(state_dict)

    def set_train_step_recorder(self, recorder):
        self.train_step_recorder = recorder

    def record_train_step(self, round_idx, local_epoch, loss, **metrics):
        if self.train_step_recorder is None:
            return
        serializable_metrics = {
            key: float(value)
            for key, value in metrics.items()
        }
        self.train_step_recorder(
            client_name=self.client_name,
            round_idx=round_idx,
            local_epoch=local_epoch,
            loss=float(loss),
            metrics=serializable_metrics,
        )

    def _warn_ddp_once(self, message, *args):
        if self._ddp_warning_emitted:
            return
        logger.warning(message, *args)
        self._ddp_warning_emitted = True

    def _resolve_brats_ddp_devices(self):
        visible_gpus = torch.cuda.device_count()
        if self.brats_ddp_devices is None:
            return list(range(visible_gpus))
        seen = []
        for device_id in self.brats_ddp_devices:
            idx = int(device_id)
            if idx not in seen and 0 <= idx < visible_gpus:
                seen.append(idx)
        return seen

    def _should_use_brats_ddp(self):
        if not self.brats_ddp:
            return False
        if self.client_name != "BraTS" and not self.ddp_force_enabled:
            return False
        if self.parallel:
            self._warn_ddp_once(
                "Client DDP is disabled because args.parallel=True. "
                "Use client DDP in serial-client mode (without --parallel)."
            )
            return False
        if not torch.cuda.is_available():
            self._warn_ddp_once("BraTS DDP requested but CUDA is not available. Falling back to single GPU.")
            return False
        device_ids = self._resolve_brats_ddp_devices()
        if len(device_ids) < 2:
            self._warn_ddp_once(
                "BraTS DDP requested but fewer than 2 usable GPUs were found (got %d). Falling back to single GPU.",
                len(device_ids),
            )
            return False
        return True

    def _evaluate_split_with_brats_ddp(self, split):
        device_ids = self._resolve_brats_ddp_devices()
        world_size = len(device_ids)
        result_queue = mp.get_context("spawn").SimpleQueue()
        worker_config = {
            "args": copy.deepcopy(self.args),
            "seed": self.seed,
            "backend": self.brats_ddp_backend,
            "init_method": f"tcp://127.0.0.1:{_find_free_port()}",
            "device_ids": device_ids,
            "client_name": self.client_name,
            "split": split,
            "model_state": self.get_model_state(),
        }
        mp.spawn(
            _client_eval_ddp_worker,
            args=(world_size, worker_config, result_queue),
            nprocs=world_size,
            join=True,
        )
        return _deserialize_queue_payload(result_queue.get())

    def train(self, round_idx, client_idx):
        raise NotImplementedError

    def _evaluate_loader(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y in dataloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                preds = torch.argmax(logits, dim=1)
                batch_size = y.shape[0]

                total_loss += loss.item() * batch_size
                total_correct += (preds == y).sum().item()
                total_samples += batch_size

                all_labels.extend(y.detach().cpu().tolist())
                all_preds.extend(preds.detach().cpu().tolist())

        accuracy = total_correct / max(total_samples, 1)
        macro_f1 = (
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
            if total_samples > 0
            else 0.0
        )
        avg_loss = total_loss / max(total_samples, 1)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "num_samples": total_samples,
        }

    def evaluate_split(self, split):
        if self._should_use_brats_ddp():
            try:
                return self._evaluate_split_with_brats_ddp(split)
            except Exception as exc:
                self.brats_ddp = False
                self._warn_ddp_once(
                    "BraTS DDP eval failed with %s. Falling back to single-GPU evaluation.",
                    type(exc).__name__,
                )
                logger.exception("BraTS DDP evaluation failed; falling back to single GPU.")
        return self._evaluate_loader(self.load_data(split=split, shuffle=False, drop_last=False))

    def train_metrics(self):
        return self.evaluate_split("train")

    def val_metrics(self):
        return self.evaluate_split("val")

    def test_metrics(self):
        return self.evaluate_split("test")
