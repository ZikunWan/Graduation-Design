import copy
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import f1_score
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


def _clienttgp_ddp_worker(rank, world_size, worker_config):
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
        client = ClientFedTGP(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_state(worker_config["client_state"])
        client.activate()
        client.set_parameters({"global_prototypes": worker_config["global_prototypes"]})
        client.model = DDP(
            client.model,
            device_ids=[device_id],
            output_device=device_id,
            broadcast_buffers=False,
        )

        train_sampler = DistributedSampler(
            client.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        trainloader = _build_ddp_dataloader(client, client.train_dataset, train_sampler, drop_last=False)
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
                train_sampler.set_epoch(worker_config["round_idx"] * max(client.local_epochs, 1) + local_epoch)
                for x, y in trainloader:
                    x = client._move_to_device(x)
                    y = y.to(client.device, non_blocking=client.pin_memory)

                    with client.autocast_context():
                        logits, fused_feature = client.model(x, return_feature=True)
                        cls_loss = client.loss_fn(logits, y)
                        proto_loss = client._prototype_alignment_loss(fused_feature, y)
                        loss = cls_loss + client.proto_lambda * proto_loss

                    client.backward_step(loss)

                    loss_stats = torch.tensor(
                        [loss.detach(), cls_loss.detach(), proto_loss.detach()],
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
                                "proto_loss": float(loss_stats[2].item()),
                            }
                        )
                        pbar.set_postfix(
                            loss=f"{loss_stats[0].item():.4f}",
                            cls=f"{loss_stats[1].item():.4f}",
                            proto=f"{loss_stats[2].item():.4f}",
                        )
                    pbar.update(1)
        finally:
            pbar.close()

        collect_sampler = DistributedSampler(
            client.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        collect_loader = _build_ddp_dataloader(client, client.train_dataset, collect_sampler, drop_last=False)
        feature_sums = torch.zeros(client.num_classes, client.model.module.backbone_dim, device=client.device, dtype=torch.float32)
        feature_counts = torch.zeros(client.num_classes, device=client.device, dtype=torch.float32)

        client.model.eval()
        with torch.no_grad():
            for x, y in collect_loader:
                x = client._move_to_device(x)
                y = y.to(client.device, non_blocking=client.pin_memory)
                with client.autocast_context():
                    _, fused_feature = client.model(x, return_feature=True)

                detached_feature = fused_feature.detach()
                for class_index in range(client.num_classes):
                    class_mask = y == class_index
                    if class_mask.any():
                        feature_sums[class_index] += detached_feature[class_mask].sum(dim=0)
                        feature_counts[class_index] += class_mask.sum().to(dtype=torch.float32)

        dist.all_reduce(feature_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_counts, op=dist.ReduceOp.SUM)

        if rank == 0:
            local_prototypes = {}
            for class_index in range(client.num_classes):
                count = feature_counts[class_index].item()
                if count <= 0:
                    continue
                local_prototypes[int(class_index)] = (feature_sums[class_index] / count).detach().cpu().clone()
            torch.save(
                {
                    "model": _clone_to_cpu(client.model.module.state_dict()),
                    "optimizer": _clone_to_cpu(client.optimizer.state_dict()),
                    "scaler": _clone_to_cpu(client.grad_scaler.state_dict()),
                    "local_prototypes": local_prototypes,
                    "step_logs": step_logs,
                },
                worker_config["result_path"],
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ClientFedTGP(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.proto_lambda = args.proto_lambda
        self.global_prototypes = None
        self.local_prototypes = {}

    def set_parameters(self, payload):
        if payload is None:
            self.global_prototypes = None
            return

        global_prototypes = payload.get("global_prototypes")
        if global_prototypes is None:
            self.global_prototypes = None
        else:
            self.global_prototypes = {
                int(label): proto.detach().cpu().clone()
                for label, proto in global_prototypes.items()
            }

    def _prototype_alignment_loss(self, fused_feature, labels):
        if not self.global_prototypes:
            return fused_feature.new_tensor(0.0)

        selected_indices = []
        target_prototypes = []
        for index, label in enumerate(labels.detach().cpu().tolist()):
            if label in self.global_prototypes:
                selected_indices.append(index)
                target_prototypes.append(self.global_prototypes[label].to(self.device))

        if not selected_indices:
            return fused_feature.new_tensor(0.0)

        selected_feature = fused_feature[selected_indices]
        target_prototypes = torch.stack(target_prototypes, dim=0)
        return F.mse_loss(selected_feature, target_prototypes)

    def collect_local_prototypes(self):
        trainloader = self.load_train_data(shuffle=False)
        self.model.eval()
        feature_sums = torch.zeros(self.num_classes, self.model.backbone_dim, device=self.device, dtype=torch.float32)
        feature_counts = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for x, y in trainloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    _, fused_feature = self.model(x, return_feature=True)

                detached_feature = fused_feature.detach()
                for class_index in range(self.num_classes):
                    class_mask = y == class_index
                    if class_mask.any():
                        feature_sums[class_index] += detached_feature[class_mask].sum(dim=0)
                        feature_counts[class_index] += class_mask.sum().to(dtype=torch.float32)

        self.local_prototypes = {}
        feature_sums = feature_sums.detach().cpu()
        feature_counts = feature_counts.detach().cpu()
        for class_index in range(self.num_classes):
            count = feature_counts[class_index].item()
            if count <= 0:
                continue
            self.local_prototypes[int(class_index)] = (feature_sums[class_index] / count).clone()
        return self.local_prototypes

    def get_prototype_payload(self):
        return {
            int(label): prototype.detach().cpu().clone()
            for label, prototype in self.local_prototypes.items()
        }

    def _build_prototype_logits(self, fused_feature):
        # Use a dtype-safe floor value so fp16 autocast does not overflow.
        floor_value = torch.finfo(fused_feature.dtype).min
        logits = fused_feature.new_full((fused_feature.size(0), self.num_classes), floor_value)
        if not self.global_prototypes:
            return logits

        for label, prototype in self.global_prototypes.items():
            prototype = prototype.to(self.device).unsqueeze(0)
            distance = torch.sum((fused_feature - prototype) ** 2, dim=1)
            logits[:, int(label)] = -distance
        return logits

    def _train_single_process(self, round_idx, client_idx):
        trainloader = self.load_train_data()
        self.model.train()

        total_steps = self.local_epochs * len(trainloader)
        with tqdm(total=total_steps, desc=f"R{round_idx} | {self.client_name}", position=client_idx, leave=False) as pbar:
            for local_epoch in range(self.local_epochs):
                for x, y in trainloader:
                    x = self._move_to_device(x)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    with self.autocast_context():
                        logits, fused_feature = self.model(x, return_feature=True)
                        cls_loss = self.loss_fn(logits, y)
                        proto_loss = self._prototype_alignment_loss(fused_feature, y)
                        loss = cls_loss + self.proto_lambda * proto_loss

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        proto_loss=proto_loss.item(),
                    )

                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        cls=f"{cls_loss.item():.4f}",
                        proto=f"{proto_loss.item():.4f}",
                    )
                    pbar.update(1)

        self.collect_local_prototypes()
        return self.local_prototypes

    def _train_with_brats_ddp(self, round_idx, client_idx):
        device_ids = self._resolve_brats_ddp_devices()
        world_size = len(device_ids)
        result_path = _create_ddp_result_path(f"{self.client_name.lower()}_tgp_ddp")
        worker_config = {
            "args": copy.deepcopy(self.args),
            "seed": self.seed,
            "round_idx": round_idx,
            "client_idx": int(client_idx),
            "backend": self.brats_ddp_backend,
            "init_method": f"tcp://127.0.0.1:{_find_free_port()}",
            "device_ids": device_ids,
            "client_name": self.client_name,
            "client_state": self.get_state(),
            "global_prototypes": {
                int(label): prototype.detach().cpu().clone()
                for label, prototype in (self.global_prototypes or {}).items()
            },
            "result_path": result_path,
        }
        mp.spawn(
            _clienttgp_ddp_worker,
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
        self.local_prototypes = {
            int(label): prototype.detach().cpu().clone()
            for label, prototype in result.get("local_prototypes", {}).items()
        }
        for step_log in result.get("step_logs", []):
            self.record_train_step(
                round_idx=round_idx,
                local_epoch=int(step_log["local_epoch"]),
                loss=float(step_log["loss"]),
                cls_loss=float(step_log["cls_loss"]),
                proto_loss=float(step_log["proto_loss"]),
            )
        return self.local_prototypes

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

    def evaluate_split(self, split):
        if not self.global_prototypes:
            return super().evaluate_split(split)

        dataloader = self.load_data(split=split, shuffle=False, drop_last=False)
        self.model.eval()

        total_loss = 0.0
        total_loss_samples = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        available_labels = set(self.global_prototypes.keys())

        with torch.no_grad():
            for x, y in dataloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    _, fused_feature = self.model(x, return_feature=True)
                    logits = self._build_prototype_logits(fused_feature)
                    preds = torch.argmax(logits, dim=1)
                batch_size = y.shape[0]

                valid_indices = [index for index, label in enumerate(y.detach().cpu().tolist()) if label in available_labels]
                if valid_indices:
                    valid_logits = logits[valid_indices]
                    valid_labels = y[valid_indices]
                    loss = self.loss_fn(valid_logits, valid_labels)
                    total_loss += loss.item() * len(valid_indices)
                    total_loss_samples += len(valid_indices)

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
        avg_loss = total_loss / max(total_loss_samples, 1)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "num_samples": total_samples,
        }
