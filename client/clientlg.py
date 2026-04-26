import copy
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


def _clientlg_ddp_worker(rank, world_size, worker_config):
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
        client = ClientLGFedAvg(args, client_name=worker_config["client_name"], device=f"cuda:{device_id}")
        client.load_state(worker_config["client_state"])
        client.activate()
        client.set_parameters({"classifier_state": worker_config["classifier_state"]})
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
                        loss = client.loss_fn(logits, y)

                    client.backward_step(loss)

                    loss_stats = torch.tensor([loss.detach()], device=client.device, dtype=torch.float32)
                    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
                    loss_stats /= float(world_size)
                    if rank == 0:
                        step_logs.append(
                            {
                                "local_epoch": int(local_epoch),
                                "loss": float(loss_stats[0].item()),
                            }
                        )
                        pbar.set_postfix(loss=f"{loss_stats[0].item():.4f}")
                    pbar.update(1)
        finally:
            pbar.close()

        if rank == 0:
            torch.save(
                {
                    "model": _clone_to_cpu(client.model.module.state_dict()),
                    "optimizer": _clone_to_cpu(client.optimizer.state_dict()),
                    "scaler": _clone_to_cpu(client.grad_scaler.state_dict()),
                    "classifier_state": _clone_to_cpu(client.model.module.classifier.state_dict()),
                    "step_logs": step_logs,
                },
                worker_config["result_path"],
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ClientLGFedAvg(Client):
    def get_classifier_state(self):
        return copy.deepcopy(self.model.classifier.state_dict())

    def load_classifier_state(self, classifier_state):
        self.model.classifier.load_state_dict(classifier_state, strict=True)

    def set_parameters(self, payload):
        if payload is None:
            return
        classifier_state = payload.get("classifier_state")
        if classifier_state is not None:
            self.load_classifier_state(classifier_state)

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
                        logits = self.model(x)
                        loss = self.loss_fn(logits, y)

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                    )

                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    pbar.update(1)

        return self.get_classifier_state()

    def _train_with_brats_ddp(self, round_idx, client_idx):
        device_ids = self._resolve_brats_ddp_devices()
        world_size = len(device_ids)
        result_path = _create_ddp_result_path(f"{self.client_name.lower()}_lg_ddp")
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
            "classifier_state": _clone_to_cpu(self.model.classifier.state_dict()),
            "result_path": result_path,
        }
        mp.spawn(
            _clientlg_ddp_worker,
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
        for step_log in result.get("step_logs", []):
            self.record_train_step(
                round_idx=round_idx,
                local_epoch=int(step_log["local_epoch"]),
                loss=float(step_log["loss"]),
            )
        return {
            key: value.detach().cpu().clone()
            for key, value in result["classifier_state"].items()
        }

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
