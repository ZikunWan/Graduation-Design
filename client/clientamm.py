from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clientbase import Client
from .clientmm import _move_optimizer_state_to_param_devices


class ClientFedAMM(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.amm_mb_lambda = float(getattr(args, "amm_mb_lambda"))
        self.amm_mc_lambda = float(getattr(args, "amm_mc_lambda"))
        self.global_combo_prototypes = {}
        self.local_combo_prototypes = {}
        self.local_combo_counts = {}

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
            self.global_combo_prototypes = {}
            return
        combo_prototypes = payload.get("global_combo_prototypes")
        self.global_combo_prototypes = {
            (int(combo_id), int(label)): prototype.detach().cpu().clone()
            for (combo_id, label), prototype in (combo_prototypes or {}).items()
        }

    def _mask_to_combo_id(self, modality_mask_row):
        combo_id = 0
        if torch.is_tensor(modality_mask_row):
            modality_mask_row = modality_mask_row.detach().cpu().tolist()
        for bit_index, value in enumerate(modality_mask_row):
            if value > 0:
                combo_id |= 1 << bit_index
        return int(combo_id)

    def _model_module(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _move_to_device(self, batch):
        if not isinstance(batch, dict) or "modalities" not in batch:
            return super()._move_to_device(batch)

        model = self._model_module()
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

    def _modality_balance_loss(self, fused_feature, modality_features, modality_mask, labels):
        active_modality_count = modality_mask.sum(dim=1)
        multimodal_sample_mask = active_modality_count > 1
        if not multimodal_sample_mask.any():
            return fused_feature.new_tensor(0.0)

        model = self._model_module()
        if not hasattr(model, "project_amm_unimodal_embeddings"):
            raise RuntimeError("FedAMM modality-balance loss requires AMMModel.")

        losses = []
        for label in labels.unique(sorted=True).tolist():
            label_mask = labels == int(label)
            multimodal_label_mask = label_mask & multimodal_sample_mask
            if not multimodal_label_mask.any():
                continue

            multimodal_prototype = fused_feature[multimodal_label_mask].mean(dim=0)
            for modality_index in range(modality_features.size(1)):
                unimodal_label_mask = label_mask & (modality_mask[:, modality_index] > 0)
                if not unimodal_label_mask.any():
                    continue
                unimodal_prototype = modality_features[unimodal_label_mask, modality_index, :].mean(dim=0)
                projected_unimodal = model.project_amm_unimodal_embeddings(unimodal_prototype)
                losses.append(F.mse_loss(projected_unimodal, multimodal_prototype.detach()))

        if not losses:
            return fused_feature.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def _combo_alignment_loss(self, fused_feature, modality_mask, labels):
        if not self.global_combo_prototypes:
            return fused_feature.new_tensor(0.0)

        grouped = defaultdict(list)
        labels_list = labels.detach().cpu().tolist()
        modality_mask_list = modality_mask.detach().cpu().tolist()
        for feature, label, mask_row in zip(fused_feature, labels_list, modality_mask_list):
            combo_id = self._mask_to_combo_id(mask_row)
            if combo_id > 0:
                grouped[(combo_id, int(label))].append(feature)

        losses = []
        for key, features in grouped.items():
            global_prototype = self.global_combo_prototypes.get(key)
            if global_prototype is None:
                continue
            local_prototype = torch.stack(features, dim=0).mean(dim=0)
            target_prototype = global_prototype.detach().to(
                device=fused_feature.device,
                dtype=local_prototype.dtype,
            )
            losses.append(F.mse_loss(local_prototype, target_prototype))

        if not losses:
            return fused_feature.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def collect_local_prototypes(self):
        trainloader = self.load_train_data(shuffle=False)
        self.model.eval()
        grouped_sums = {}
        grouped_counts = defaultdict(int)

        with torch.no_grad():
            for x, y in trainloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    outputs = self.model(x, return_dict=True)
                fused_feature = outputs["fused_feature"].detach().cpu()
                modality_mask = outputs["modality_mask"].detach().cpu().tolist()
                labels = y.detach().cpu().tolist()
                for feature, label, mask_row in zip(fused_feature, labels, modality_mask):
                    combo_id = self._mask_to_combo_id(mask_row)
                    if combo_id <= 0:
                        continue
                    key = (combo_id, int(label))
                    if key not in grouped_sums:
                        grouped_sums[key] = torch.zeros_like(feature)
                    grouped_sums[key] += feature
                    grouped_counts[key] += 1

        self.local_combo_prototypes = {
            key: feature_sum / max(grouped_counts[key], 1)
            for key, feature_sum in grouped_sums.items()
        }
        self.local_combo_counts = dict(grouped_counts)
        return self.local_combo_prototypes

    def get_prototype_payload(self):
        return {
            "combo_prototypes": {
                key: prototype.detach().cpu().clone()
                for key, prototype in self.local_combo_prototypes.items()
            },
            "combo_counts": dict(self.local_combo_counts),
        }

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            self.brats_ddp = False
            self._warn_ddp_once("FedAMM client uses single-process training in this implementation.")

        trainloader = self.load_train_data()
        self.model.train()
        total_steps = self.local_epochs * len(trainloader)

        with tqdm(total=total_steps, desc=f"R{round_idx} | {self.client_name}", position=client_idx, leave=False) as pbar:
            for local_epoch in range(self.local_epochs):
                for x, y in trainloader:
                    x = self._move_to_device(x)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    with self.autocast_context():
                        outputs = self.model(x, return_dict=True)
                        logits = outputs["logits"]
                        fused_feature = outputs["fused_feature"]
                        modality_features = outputs["modality_features"]
                        modality_mask = outputs["modality_mask"]

                        cls_loss = self.loss_fn(logits, y)
                        mb_loss = self._modality_balance_loss(fused_feature, modality_features, modality_mask, y)
                        mc_loss = self._combo_alignment_loss(fused_feature, modality_mask, y)
                        loss = cls_loss + self.amm_mb_lambda * mb_loss + self.amm_mc_lambda * mc_loss

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        mb_loss=mb_loss.item(),
                        mc_loss=mc_loss.item(),
                    )
                    pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", mb=f"{mb_loss.item():.4f}", mc=f"{mc_loss.item():.4f}")
                    pbar.update(1)

        self.collect_local_prototypes()
        return self.local_combo_prototypes
