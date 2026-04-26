from collections import defaultdict

import torch
from tqdm import tqdm

from .clientbase import Client


def _move_value_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_param_devices(optimizer):
    for parameter, state in optimizer.state.items():
        if not torch.is_tensor(parameter):
            continue
        for key, value in list(state.items()):
            state[key] = _move_value_to_device(value, parameter.device)


class ClientFedMM(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.loss_beta = float(getattr(args, "fedmm_beta", 0.25))
        self.loss_alpha = float(getattr(args, "fedmm_alpha", 0.05))
        self.loss_transition_round = float(getattr(args, "fedmm_t0", 30.0))
        self.global_modality_prototypes = {}
        self.local_modality_prototypes = {}

    def _model_parallel_devices(self):
        if self.device.type != "cuda":
            return []
        if self.brats_ddp_devices is None:
            return []
        device_ids = self._resolve_brats_ddp_devices()
        if len(device_ids) <= 1:
            return []
        return [torch.device(f"cuda:{device_id}") for device_id in device_ids]

    def _should_use_model_parallel(self):
        return len(self._model_parallel_devices()) > 1

    def _should_use_brats_ddp(self):
        return False

    def activate(self):
        if not self.enable_model_offload or self.is_model_materialized:
            if self._should_use_model_parallel() and hasattr(self.model, "configure_model_parallel"):
                self.model.configure_model_parallel(self._model_parallel_devices())
                _move_optimizer_state_to_param_devices(self.optimizer)
            return

        if self._should_use_model_parallel() and hasattr(self.model, "configure_model_parallel"):
            self.model.configure_model_parallel(self._model_parallel_devices())
            _move_optimizer_state_to_param_devices(self.optimizer)
            self.is_model_materialized = True
            return

        super().activate()

    def offload(self):
        if not self.enable_model_offload or not self.is_model_materialized:
            return
        cpu_device = torch.device("cpu")
        self.model.to(cpu_device)
        _move_optimizer_state_to_param_devices(self.optimizer)
        self.is_model_materialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_parameters(self, payload):
        if payload is None:
            self.global_modality_prototypes = {}
            return
        prototypes = payload.get("global_modality_prototypes")
        self.global_modality_prototypes = {
            (str(modality), int(label)): prototype.detach().cpu().clone()
            for (modality, label), prototype in (prototypes or {}).items()
        }

    def _dynamic_lambda(self, round_idx):
        t = torch.tensor(float(round_idx), dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.loss_alpha * (t - self.loss_transition_round))

    def _model_modalities(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        return model.modalities

    def _move_to_device(self, batch):
        if not isinstance(batch, dict) or "modalities" not in batch:
            return super()._move_to_device(batch)

        model = self.model.module if hasattr(self.model, "module") else self.model
        modality_devices = getattr(model, "modality_devices", {})
        moved_batch = {}
        for key, value in batch.items():
            if key == "modalities":
                moved_batch[key] = {
                    modality: tensor.to(
                        modality_devices.get(modality, self.device),
                        non_blocking=self.pin_memory,
                    )
                    for modality, tensor in value.items()
                }
            elif torch.is_tensor(value):
                moved_batch[key] = value.to(self.device, non_blocking=self.pin_memory)
            else:
                moved_batch[key] = super()._move_to_device(value)
        return moved_batch

    def _prototype_l2_loss(self, modality_prototypes, modality_mask, labels):
        if not self.global_modality_prototypes:
            return modality_prototypes.new_tensor(0.0)

        losses = []
        for modality_index, modality in enumerate(self._model_modalities()):
            active_mask = modality_mask[:, modality_index] > 0
            if not active_mask.any():
                continue
            for row_index, label in enumerate(labels.detach().cpu().tolist()):
                if not bool(active_mask[row_index]):
                    continue
                global_prototype = self.global_modality_prototypes.get((modality, int(label)))
                if global_prototype is None:
                    continue
                local_prototype = modality_prototypes[row_index, modality_index]
                target_prototype = global_prototype.to(self.device, dtype=local_prototype.dtype)
                losses.append(torch.sum((local_prototype - target_prototype) ** 2))

        if not losses:
            return modality_prototypes.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def _fedmm_loss(self, cls_loss, proto_loss, round_idx, prototype_dim):
        lambda_t = self._dynamic_lambda(round_idx).to(device=cls_loss.device, dtype=cls_loss.dtype)
        denominator = max(float(prototype_dim), 1e-12)
        loss = (self.loss_beta * lambda_t / denominator) * proto_loss + (1.0 - lambda_t) * cls_loss
        return loss, lambda_t

    def collect_local_prototypes(self):
        trainloader = self.load_train_data(shuffle=False)
        self.model.eval()
        grouped = defaultdict(list)

        with torch.no_grad():
            for x, y in trainloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    outputs = self.model(x, return_dict=True, return_prototype=True)
                modality_prototypes = outputs["modality_prototypes"].detach().cpu()
                modality_mask = outputs["modality_mask"].detach().cpu()
                labels = y.detach().cpu().tolist()

                for row_index, label in enumerate(labels):
                    for modality_index, modality in enumerate(self._model_modalities()):
                        if modality_mask[row_index, modality_index] <= 0:
                            continue
                        key = (modality, int(label))
                        grouped[key].append(modality_prototypes[row_index, modality_index])

        self.local_modality_prototypes = {
            key: torch.stack(features, dim=0).mean(dim=0)
            for key, features in grouped.items()
        }
        return self.local_modality_prototypes

    def get_prototype_payload(self):
        return {
            "modality_prototypes": {
                key: prototype.detach().cpu().clone()
                for key, prototype in self.local_modality_prototypes.items()
            },
        }

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
                        outputs = self.model(x, return_dict=True, return_prototype=True)
                        logits = outputs["logits"]
                        modality_prototypes = outputs["modality_prototypes"]
                        modality_mask = outputs["modality_mask"]
                        cls_loss = self.loss_fn(logits, y)
                        proto_loss = self._prototype_l2_loss(modality_prototypes, modality_mask, y)
                        prototype_dim = modality_prototypes.shape[-1]
                        loss, lambda_t = self._fedmm_loss(cls_loss, proto_loss, round_idx, prototype_dim)

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        proto_loss=proto_loss.item(),
                        fedmm_lambda=lambda_t.item(),
                    )
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        cls=f"{cls_loss.item():.4f}",
                        proto=f"{proto_loss.item():.4f}",
                        lam=f"{lambda_t.item():.3f}",
                    )
                    pbar.update(1)

        self.collect_local_prototypes()
        return self.local_modality_prototypes

    def train(self, round_idx, client_idx):
        return self._train_single_process(round_idx, client_idx)
