from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clientbase import Client


class ClientFedAMM(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.amm_mb_lambda = float(getattr(args, "amm_mb_lambda"))
        self.amm_mc_lambda = float(getattr(args, "amm_mc_lambda"))
        self.global_combo_prototypes = {}
        self.local_combo_prototypes = {}
        self.local_combo_counts = {}

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
        for bit_index, value in enumerate(modality_mask_row.detach().cpu().tolist()):
            if value > 0:
                combo_id |= 1 << bit_index
        return int(combo_id)

    def _compute_proto_means(self, features, labels, sample_mask=None):
        proto_map = {}
        for label in labels.unique(sorted=True).tolist():
            label_mask = labels == int(label)
            if sample_mask is not None:
                label_mask = label_mask & sample_mask
            if label_mask.any():
                proto_map[int(label)] = features[label_mask].mean(dim=0)
        return proto_map

    def _modality_balance_loss(self, fused_feature, modality_features, modality_mask, labels):
        teacher_prototypes = self._compute_proto_means(fused_feature, labels)
        losses = []

        for modality_index in range(modality_features.size(1)):
            active_mask = modality_mask[:, modality_index] > 0
            if not active_mask.any():
                continue
            student_prototypes = self._compute_proto_means(
                modality_features[:, modality_index, :],
                labels,
                sample_mask=active_mask,
            )
            for label, teacher_prototype in teacher_prototypes.items():
                student_prototype = student_prototypes.get(label)
                if student_prototype is not None:
                    losses.append(F.mse_loss(student_prototype, teacher_prototype.detach()))

        if not losses:
            return fused_feature.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def _combo_alignment_loss(self, fused_feature, modality_mask, labels):
        if not self.global_combo_prototypes:
            return fused_feature.new_tensor(0.0)

        grouped = defaultdict(list)
        for feature, label, mask_row in zip(fused_feature, labels.detach().cpu().tolist(), modality_mask):
            combo_id = self._mask_to_combo_id(mask_row)
            if combo_id > 0:
                grouped[(combo_id, int(label))].append(feature)

        losses = []
        for key, features in grouped.items():
            global_prototype = self.global_combo_prototypes.get(key)
            if global_prototype is None:
                continue
            local_prototype = torch.stack(features, dim=0).mean(dim=0)
            losses.append(F.mse_loss(local_prototype, global_prototype.to(self.device)))

        if not losses:
            return fused_feature.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def collect_local_prototypes(self):
        trainloader = self.load_train_data(shuffle=False)
        self.model.eval()
        grouped_features = defaultdict(list)
        grouped_counts = defaultdict(int)

        with torch.no_grad():
            for x, y in trainloader:
                x = self._move_to_device(x)
                y = y.to(self.device, non_blocking=self.pin_memory)
                with self.autocast_context():
                    outputs = self.model(x, return_dict=True)
                fused_feature = outputs["fused_feature"].detach().cpu()
                modality_mask = outputs["modality_mask"].detach().cpu()
                for feature, label, mask_row in zip(fused_feature, y.detach().cpu().tolist(), modality_mask):
                    combo_id = self._mask_to_combo_id(mask_row)
                    if combo_id <= 0:
                        continue
                    key = (combo_id, int(label))
                    grouped_features[key].append(feature)
                    grouped_counts[key] += 1

        self.local_combo_prototypes = {
            key: torch.stack(features, dim=0).mean(dim=0)
            for key, features in grouped_features.items()
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
