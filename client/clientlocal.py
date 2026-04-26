import copy
import logging
import queue
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from .clientbase import (
    Client,
    _build_ddp_dataloader,
    _clone_to_cpu,
    _configure_ddp_worker_output,
    _deserialize_queue_payload,
    _find_free_port,
    _serialize_queue_payload,
    move_optimizer_state,
)

logger = logging.getLogger(__name__)


def _evaluate_loader_ddp(client, dataloader, world_size):
    client.model.eval()
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
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": total_samples_value,
    }


def _clientlocal_ddp_epoch_worker(rank, world_size, worker_config, result_queue):
    _configure_ddp_worker_output(rank)
    device_ids = worker_config["device_ids"]
    device_id = device_ids[rank]

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
        client = ClientLocal(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_state(worker_config["client_state"])
        client.activate()
        client.model = DDP(
            client.model,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
        )

        sampler = DistributedSampler(
            client.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        sampler.set_epoch(int(worker_config["round_idx"]))
        trainloader = _build_ddp_dataloader(client, client.train_dataset, sampler, drop_last=False)

        step_logs = []
        pbar = tqdm(
            total=len(trainloader),
            desc=f"R{worker_config['round_idx']} | {client.client_name}(DDP)",
            position=int(worker_config.get("client_idx", 0)),
            leave=False,
            disable=rank != 0,
        )
        try:
            for x, y in trainloader:
                x = client._move_to_device(x)
                y = y.to(client.device, non_blocking=client.pin_memory)

                with client.autocast_context():
                    logits = client.model(x)
                    loss = client.loss_fn(logits, y)

                client.backward_step(loss)

                loss_stats = torch.tensor([loss.detach()], device=client.device, dtype=torch.float32)
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                loss_stats /= float(world_size)
                if rank == 0:
                    step_logs.append(
                        {
                            "loss": float(loss_stats[0].item()),
                        }
                    )
                    pbar.set_postfix(loss=f"{loss_stats[0].item():.4f}")
                pbar.update(1)
        finally:
            pbar.close()

        if rank == 0:
            result_queue.put(
                _serialize_queue_payload(
                    {
                        "model": _clone_to_cpu(client.model.module.state_dict()),
                        "optimizer": _clone_to_cpu(client.optimizer.state_dict()),
                        "scaler": _clone_to_cpu(client.grad_scaler.state_dict()),
                        "step_logs": step_logs,
                    }
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _clientlocal_ddp_persistent_worker(rank, world_size, worker_config, command_queue, result_queue):
    _configure_ddp_worker_output(rank)
    device_ids = worker_config["device_ids"]
    device_id = device_ids[rank]

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
        client = ClientLocal(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_state(worker_config["client_state"])
        client.activate()
        client.model = DDP(
            client.model,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
        )

        sampler = DistributedSampler(
            client.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        trainloader = _build_ddp_dataloader(client, client.train_dataset, sampler, drop_last=False)

        while True:
            command = None
            if rank == 0:
                command = command_queue.get()
            command_box = [command]
            dist.broadcast_object_list(command_box, src=0)
            command = command_box[0] or {}
            command_type = command.get("cmd")

            if command_type == "stop":
                break

            if command_type == "eval_split":
                split = command.get("split", "val")
                if split not in client.datasets:
                    if rank == 0:
                        result_queue.put(
                            _serialize_queue_payload(
                                {
                                    "ok": False,
                                    "error_type": "ValueError",
                                    "error_msg": f"Unsupported eval split '{split}'",
                                    "traceback": "",
                                }
                            )
                        )
                    continue

                eval_dataset = client.datasets[split]
                eval_sampler = DistributedSampler(
                    eval_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                eval_loader = _build_ddp_dataloader(client, eval_dataset, eval_sampler, drop_last=False)
                metrics = _evaluate_loader_ddp(client, eval_loader, world_size)
                if rank == 0:
                    result_queue.put(
                        _serialize_queue_payload(
                            {
                                "ok": True,
                                "metrics": metrics,
                            }
                        )
                    )
                continue

            if command_type != "train_round":
                if rank == 0:
                    result_queue.put(
                        _serialize_queue_payload(
                            {
                                "ok": False,
                                "error_type": "ValueError",
                                "error_msg": f"Unsupported command '{command_type}'",
                                "traceback": "",
                            }
                        )
                    )
                continue

            round_idx = int(command["round_idx"])
            client_idx = int(command.get("client_idx", 0))
            sampler.set_epoch(round_idx)

            step_logs = []
            pbar = tqdm(
                total=len(trainloader),
                desc=f"R{round_idx} | {client.client_name}(DDP)",
                position=client_idx,
                leave=False,
                disable=rank != 0,
            )
            try:
                for x, y in trainloader:
                    x = client._move_to_device(x)
                    y = y.to(client.device, non_blocking=client.pin_memory)

                    with client.autocast_context():
                        logits = client.model(x)
                        loss = client.loss_fn(logits, y)

                    client.backward_step(loss)

                    loss_stats = torch.tensor([loss.detach()], device=client.device, dtype=torch.float32)
                    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                    loss_stats /= float(world_size)
                    if rank == 0:
                        step_logs.append({"loss": float(loss_stats[0].item())})
                        pbar.set_postfix(loss=f"{loss_stats[0].item():.4f}")
                    pbar.update(1)
            finally:
                pbar.close()

            if rank == 0:
                result_queue.put(
                    _serialize_queue_payload(
                        {
                            "ok": True,
                            "model": _clone_to_cpu(client.model.module.state_dict()),
                            "optimizer": _clone_to_cpu(client.optimizer.state_dict()),
                            "scaler": _clone_to_cpu(client.grad_scaler.state_dict()),
                            "step_logs": step_logs,
                        }
                    )
                )
    except Exception as exc:
        if rank == 0:
            result_queue.put(
                _serialize_queue_payload(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ClientLocal(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.early_stopping_patience = getattr(args, "early_stopping_patience", 5)
        self.early_stopping_min_delta = getattr(args, "early_stopping_min_delta", 0.0)
        self.early_stopping_enabled = self.early_stopping_patience > 0 and self.val_samples > 0
        self.early_stopped = False
        self.early_stopped_round = None
        self.best_val_accuracy = None
        self.best_model_state = None
        self.best_optimizer_state = None
        self.best_round = None
        self.no_improvement_rounds = 0
        self.last_val_metrics = None
        self._ddp_process_context = None
        self._ddp_command_queue = None
        self._ddp_result_queue = None

        if self.val_samples == 0:
            logger.warning(
                "Client %s has no validation samples; local early stopping is disabled for this client.",
                self.client_name,
            )

    def _stop_brats_ddp_persistent_session(self):
        if self._ddp_process_context is None:
            return
        try:
            if self._ddp_command_queue is not None:
                self._ddp_command_queue.put({"cmd": "stop"})
        except Exception:
            logger.exception("Failed to send stop command to BraTS DDP workers.")

        try:
            while not self._ddp_process_context.join(timeout=5):
                pass
        except Exception:
            logger.exception("BraTS DDP worker shutdown raised an exception.")
        finally:
            self._ddp_process_context = None
            if self._ddp_command_queue is not None:
                self._ddp_command_queue.close()
                self._ddp_command_queue.join_thread()
            if self._ddp_result_queue is not None:
                self._ddp_result_queue.close()
                self._ddp_result_queue.join_thread()
            self._ddp_command_queue = None
            self._ddp_result_queue = None

    def shutdown_ddp_workers(self):
        self._stop_brats_ddp_persistent_session()

    def _ensure_brats_ddp_persistent_session(self):
        if self._ddp_process_context is not None:
            return

        device_ids = self._resolve_brats_ddp_devices()
        world_size = len(device_ids)
        spawn_ctx = mp.get_context("spawn")
        self._ddp_command_queue = spawn_ctx.Queue()
        self._ddp_result_queue = spawn_ctx.Queue()
        worker_config = {
            "args": copy.deepcopy(self.args),
            "seed": self.seed,
            "backend": self.brats_ddp_backend,
            "init_method": f"tcp://127.0.0.1:{_find_free_port()}",
            "device_ids": device_ids,
            "client_name": self.client_name,
            "client_state": self.get_state(),
        }
        self._ddp_process_context = mp.spawn(
            _clientlocal_ddp_persistent_worker,
            args=(world_size, worker_config, self._ddp_command_queue, self._ddp_result_queue),
            nprocs=world_size,
            join=False,
        )

    def _run_one_epoch_single(self, round_idx, client_idx):
        trainloader = self.load_train_data()
        self.model.train()

        total_steps = len(trainloader)
        with tqdm(total=total_steps, desc=f"R{round_idx} | {self.client_name}", position=client_idx, leave=False) as pbar:
            for x, y in trainloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)

                with self.autocast_context():
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                self.backward_step(loss)
                self.record_train_step(
                    round_idx=round_idx,
                    local_epoch=0,
                    loss=loss.item(),
                )

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

    def _run_one_epoch_brats_ddp(self, round_idx, client_idx):
        self._ensure_brats_ddp_persistent_session()
        self._ddp_command_queue.put(
            {
                "cmd": "train_round",
                "round_idx": int(round_idx),
                "client_idx": int(client_idx),
            }
        )
        result = self._wait_brats_ddp_result()
        if not result.get("ok", False):
            error_type = result.get("error_type", "RuntimeError")
            error_msg = result.get("error_msg", "")
            tb = result.get("traceback", "")
            raise RuntimeError(f"{error_type}: {error_msg}\n{tb}".strip())

        self.load_state(
            {
                "model": result["model"],
                "optimizer": result["optimizer"],
                "scaler": result["scaler"],
            }
        )
        for step_log in result.get("step_logs", []):
            self.record_train_step(
                round_idx=round_idx,
                local_epoch=0,
                loss=float(step_log["loss"]),
            )

    def _wait_brats_ddp_result(self):
        result_payload = None
        deadline = time.time() + 60 * 60
        while result_payload is None:
            try:
                result_payload = self._ddp_result_queue.get(timeout=5.0)
            except queue.Empty:
                if time.time() >= deadline:
                    raise TimeoutError("Timed out waiting for BraTS DDP worker result.")
                if self._ddp_process_context is not None:
                    try:
                        if self._ddp_process_context.join(timeout=0):
                            raise RuntimeError("BraTS DDP workers exited unexpectedly.")
                    except Exception as exc:
                        raise RuntimeError("BraTS DDP workers failed while waiting for result.") from exc
        return _deserialize_queue_payload(result_payload)

    def _run_one_epoch(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            try:
                self._run_one_epoch_brats_ddp(round_idx, client_idx)
                return
            except Exception as exc:
                self._stop_brats_ddp_persistent_session()
                self.brats_ddp = False
                self._warn_ddp_once(
                    "BraTS DDP failed with %s. Falling back to single-GPU training.",
                    type(exc).__name__,
                )
                logger.exception("BraTS DDP training failed; falling back to single GPU.")
        self._run_one_epoch_single(round_idx, client_idx)

    def _evaluate_split_brats_ddp_persistent(self, split):
        self._ensure_brats_ddp_persistent_session()
        self._ddp_command_queue.put(
            {
                "cmd": "eval_split",
                "split": split,
            }
        )
        result = self._wait_brats_ddp_result()
        if not result.get("ok", False):
            error_type = result.get("error_type", "RuntimeError")
            error_msg = result.get("error_msg", "")
            tb = result.get("traceback", "")
            raise RuntimeError(f"{error_type}: {error_msg}\n{tb}".strip())
        return result["metrics"]

    def evaluate_split(self, split):
        if self._ddp_process_context is not None and self._should_use_brats_ddp():
            try:
                return self._evaluate_split_brats_ddp_persistent(split)
            except Exception as exc:
                self._stop_brats_ddp_persistent_session()
                self.brats_ddp = False
                self._warn_ddp_once(
                    "BraTS DDP eval failed with %s. Falling back to single-GPU evaluation.",
                    type(exc).__name__,
                )
                logger.exception("BraTS DDP evaluation failed; falling back to single GPU.")
        return super().evaluate_split(split)

    def _update_early_stopping(self, round_idx):
        if self.val_samples == 0:
            self.last_val_metrics = None
            return None

        self.last_val_metrics = self.val_metrics()
        val_accuracy = self.last_val_metrics["accuracy"]

        improved = self.best_val_accuracy is None or val_accuracy > (self.best_val_accuracy + self.early_stopping_min_delta)
        if improved:
            self.best_val_accuracy = val_accuracy
            self.best_model_state = self.get_model_state()
            self.best_optimizer_state = _clone_to_cpu(self.optimizer.state_dict())
            self.best_round = round_idx
            self.no_improvement_rounds = 0
            return self.last_val_metrics

        if not self.early_stopping_enabled:
            return self.last_val_metrics

        self.no_improvement_rounds += 1
        if self.no_improvement_rounds >= self.early_stopping_patience:
            if self.best_model_state is not None:
                self.load_model_state(self.best_model_state)
            if self.best_optimizer_state is not None:
                self.optimizer.load_state_dict(self.best_optimizer_state)
                if self.is_model_materialized:
                    move_optimizer_state(self.optimizer, self.device)
            self.early_stopped = True
            self.early_stopped_round = round_idx
            logger.info(
                "Client %s early-stopped at round %d; best val accuracy %.6f was reached at round %d.",
                self.client_name,
                round_idx,
                self.best_val_accuracy if self.best_val_accuracy is not None else float("nan"),
                self.best_round if self.best_round is not None else -1,
            )
        return self.last_val_metrics

    def train(self, round_idx, client_idx):
        if self.early_stopped:
            return self.get_model_state()

        self._run_one_epoch(round_idx, client_idx)
        self._update_early_stopping(round_idx)
        return self.get_model_state()
