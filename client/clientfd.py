import copy
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from .clientbase import (
    Client,
    _build_ddp_dataloader,
    _clone_to_cpu,
    _configure_ddp_worker_output,
    _create_ddp_result_path,
    _find_free_port,
    _load_and_cleanup_ddp_result,
)

logger = logging.getLogger(__name__)


def _clientfd_ddp_worker(rank, world_size, worker_config):
    _configure_ddp_worker_output(rank)
    device_ids = worker_config["device_ids"]
    device_id = device_ids[rank]
    init_method = worker_config["init_method"]
    backend = worker_config["backend"]

    args = copy.deepcopy(worker_config["args"])
    args.device = f"cuda:{device_id}"
    args.parallel = False
    args.brats_ddp = False

    seed = int(worker_config["seed"]) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device_id)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    try:
        client = ClientFD(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_state(worker_config["client_state"])
        client.activate()
        client.set_parameters({"global_logits": worker_config["global_logits"]})
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

        class_logit_sums = torch.zeros(
            client.num_classes,
            client.num_classes,
            device=client.device,
            dtype=torch.float32,
        )
        class_counts = torch.zeros(client.num_classes, device=client.device, dtype=torch.float32)
        step_logs = []

        pbar = tqdm(
            total=client.local_epochs * len(trainloader),
            desc=f"R{worker_config['round_idx']} | {client.client_name}(DDP)",
            position=int(worker_config.get("client_idx", 0)),
            leave=False,
            disable=rank != 0,
        )
        try:
            for local_epoch in range(client.local_epochs):
                sampler.set_epoch(worker_config["round_idx"] * max(client.local_epochs, 1) + local_epoch)
                for x, y in trainloader:
                    x = client._move_to_device(x)
                    y = y.to(client.device, non_blocking=client.pin_memory)

                    with client.autocast_context():
                        logits = client.model(x)
                        cls_loss = client.loss_fn(logits, y)
                        distill_loss = client._distillation_loss(logits, y)
                        loss = cls_loss + client.fd_lambda * distill_loss

                    client.backward_step(loss)

                    loss_stats = torch.tensor(
                        [loss.detach(), cls_loss.detach(), distill_loss.detach()],
                        device=client.device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                    loss_stats /= float(world_size)
                    if rank == 0:
                        step_logs.append(
                            {
                                "local_epoch": int(local_epoch),
                                "loss": float(loss_stats[0].item()),
                                "cls_loss": float(loss_stats[1].item()),
                                "kd_loss": float(loss_stats[2].item()),
                            }
                        )
                        pbar.set_postfix(
                            loss=f"{loss_stats[0].item():.4f}",
                            cls=f"{loss_stats[1].item():.4f}",
                            kd=f"{loss_stats[2].item():.4f}",
                        )

                    detached_logits = logits.detach()
                    for class_index in range(client.num_classes):
                        class_mask = y == class_index
                        if class_mask.any():
                            class_logit_sums[class_index] += detached_logits[class_mask].sum(dim=0)
                            class_counts[class_index] += class_mask.sum().to(dtype=torch.float32)
                    pbar.update(1)
        finally:
            pbar.close()

        dist.all_reduce(class_logit_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)

        if rank == 0:
            local_logits = {}
            for class_index in range(client.num_classes):
                count = class_counts[class_index].item()
                if count <= 0:
                    continue
                local_logits[int(class_index)] = (
                    class_logit_sums[class_index] / count
                ).detach().cpu().clone()

            torch.save(
                {
                    "model": _clone_to_cpu(client.model.module.state_dict()),
                    "optimizer": _clone_to_cpu(client.optimizer.state_dict()),
                    "scaler": _clone_to_cpu(client.grad_scaler.state_dict()),
                    "local_logits": local_logits,
                    "step_logs": step_logs,
                },
                worker_config["result_path"],
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ClientFD(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.fd_lambda = args.fd_lambda
        self.global_logits = None
        self.local_logits = {}

    def set_parameters(self, payload):
        if payload is None:
            self.global_logits = None
            return

        if "model_state" in payload and payload["model_state"] is not None:
            self.load_model_state(payload["model_state"])

        global_logits = payload.get("global_logits")
        if global_logits is None:
            self.global_logits = None
        else:
            self.global_logits = {
                int(label): logit.detach().cpu().clone()
                for label, logit in global_logits.items()
            }

    def _distillation_loss(self, logits, labels):
        if not self.global_logits:
            return logits.new_tensor(0.0)

        selected_indices = []
        teacher_logits = []
        for index, label in enumerate(labels.detach().cpu().tolist()):
            if label in self.global_logits:
                selected_indices.append(index)
                teacher_logits.append(self.global_logits[label].to(self.device))

        if not selected_indices:
            return logits.new_tensor(0.0)

        student_logits = logits[selected_indices]
        teacher_logits = torch.stack(teacher_logits, dim=0)
        teacher_probs = F.softmax(teacher_logits, dim=1)
        return -(teacher_probs * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()

    def get_logits_payload(self):
        return {
            int(label): logit.detach().cpu().clone()
            for label, logit in self.local_logits.items()
        }

    def _train_single_process(self, round_idx, client_idx):
        trainloader = self.load_train_data()
        self.model.train()
        class_logit_sums = torch.zeros(
            self.num_classes,
            self.num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        class_counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)

        total_steps = self.local_epochs * len(trainloader)
        with tqdm(total=total_steps, desc=f"R{round_idx} | {self.client_name}", position=client_idx, leave=False) as pbar:
            for local_epoch in range(self.local_epochs):
                for x, y in trainloader:
                    x = self._move_to_device(x)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    with self.autocast_context():
                        logits = self.model(x)
                        cls_loss = self.loss_fn(logits, y)
                        distill_loss = self._distillation_loss(logits, y)
                        loss = cls_loss + self.fd_lambda * distill_loss

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        kd_loss=distill_loss.item(),
                    )

                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        cls=f"{cls_loss.item():.4f}",
                        kd=f"{distill_loss.item():.4f}",
                    )
                    pbar.update(1)

                    detached_logits = logits.detach()
                    for class_index in range(self.num_classes):
                        class_mask = y == class_index
                        if class_mask.any():
                            class_logit_sums[class_index] += detached_logits[class_mask].sum(dim=0)
                            class_counts[class_index] += class_mask.sum().to(dtype=torch.float32)

        self.local_logits = {}
        class_logit_sums = class_logit_sums.detach().cpu()
        class_counts = class_counts.detach().cpu()
        for class_index in range(self.num_classes):
            count = class_counts[class_index].item()
            if count <= 0:
                continue
            self.local_logits[int(class_index)] = (class_logit_sums[class_index] / count).clone()
        return self.local_logits

    def _train_with_brats_ddp(self, round_idx, client_idx):
        device_ids = self._resolve_brats_ddp_devices()
        world_size = len(device_ids)
        port = _find_free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        result_path = _create_ddp_result_path(f"{self.client_name.lower()}_fd_ddp")
        worker_config = {
            "args": copy.deepcopy(self.args),
            "seed": self.seed,
            "round_idx": round_idx,
            "client_idx": int(client_idx),
            "backend": self.brats_ddp_backend,
            "init_method": init_method,
            "device_ids": device_ids,
            "client_name": self.client_name,
            "client_state": self.get_state(),
            "global_logits": {
                int(label): logit.detach().cpu().clone()
                for label, logit in (self.global_logits or {}).items()
            },
            "result_path": result_path,
        }

        mp.spawn(
            _clientfd_ddp_worker,
            args=(world_size, worker_config),
            nprocs=world_size,
            join=True,
        )
        result = _load_and_cleanup_ddp_result(result_path)
        self.load_state(
            {
                "model": result["model"],
                "optimizer": result["optimizer"],
                "scaler": result["scaler"],
            }
        )
        self.local_logits = {
            int(label): logit.detach().cpu().clone()
            for label, logit in result.get("local_logits", {}).items()
        }
        for step_log in result.get("step_logs", []):
            self.record_train_step(
                round_idx=round_idx,
                local_epoch=int(step_log["local_epoch"]),
                loss=float(step_log["loss"]),
                cls_loss=float(step_log["cls_loss"]),
                kd_loss=float(step_log["kd_loss"]),
            )
        return self.local_logits

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            try:
                return self._train_with_brats_ddp(round_idx, client_idx)
            except Exception as exc:
                self.brats_ddp = False
                self._warn_ddp_once(
                    "BraTS DDP failed with %s. Falling back to single-GPU training.",
                    type(exc).__name__,
                )
                logger.exception("BraTS DDP training failed; falling back to single GPU.")
        return self._train_single_process(round_idx, client_idx)
