from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clientbase import Client


class ClientPEPSY(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.control_lambda = float(getattr(args, "pepsy_control_lambda", 1.0))
        self.contrastive_lambda = float(getattr(args, "pepsy_contrastive_lambda", 0.2))
        self.temperature = float(getattr(args, "pepsy_temperature", 0.2))
        self.global_controls = {}
        self.local_controls = {}
        self.local_control_counts = {}

    def set_parameters(self, payload):
        if payload is None:
            self.global_controls = {}
            return
        controls = payload.get("global_controls")
        self.global_controls = {
            int(combo_id): control.detach().cpu().clone()
            for combo_id, control in (controls or {}).items()
        }

        if self.global_controls and hasattr(self.model, "pattern_controls"):
            with torch.no_grad():
                for combo_id, control in self.global_controls.items():
                    if 0 <= combo_id < self.model.pattern_controls.weight.shape[0]:
                        self.model.pattern_controls.weight[combo_id].copy_(
                            control.to(self.model.pattern_controls.weight.device)
                        )

    def _control_alignment_loss(self, combo_ids, missing_controls):
        if not self.global_controls:
            return missing_controls.new_tensor(0.0)

        selected_indices = []
        target_controls = []
        for index, combo_id in enumerate(combo_ids.detach().cpu().tolist()):
            global_control = self.global_controls.get(int(combo_id))
            if global_control is not None:
                selected_indices.append(index)
                target_controls.append(global_control.to(self.device))

        if not selected_indices:
            return missing_controls.new_tensor(0.0)

        local_controls = missing_controls[selected_indices]
        target_controls = torch.stack(target_controls, dim=0)
        return F.mse_loss(local_controls, target_controls)

    def _supervised_contrastive_loss(self, features, labels):
        if features.size(0) < 2:
            return features.new_tensor(0.0)

        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.t()) / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        eye = torch.eye(features.size(0), device=features.device, dtype=positive_mask.dtype)
        positive_mask = positive_mask * (1.0 - eye)

        positives_per_row = positive_mask.sum(dim=1)
        valid_rows = positives_per_row > 0
        if not valid_rows.any():
            return features.new_tensor(0.0)

        exp_logits = torch.exp(logits) * (1.0 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_row.clamp_min(1.0)
        return -mean_log_prob_pos[valid_rows].mean()

    def collect_local_controls(self):
        trainloader = self.load_train_data(shuffle=False)
        counts = defaultdict(int)

        self.model.eval()
        with torch.no_grad():
            for x, _ in trainloader:
                x = self._move_to_device(x)
                outputs = self.model(x, return_dict=True)
                for combo_id in outputs["combo_id"].detach().cpu().tolist():
                    counts[int(combo_id)] += 1

        weight = self.model.pattern_controls.weight.detach().cpu()
        self.local_controls = {
            int(combo_id): weight[int(combo_id)].clone()
            for combo_id, count in counts.items()
            if count > 0
        }
        self.local_control_counts = dict(counts)
        return self.local_controls

    def get_control_payload(self):
        return {
            "controls": {
                int(combo_id): control.detach().cpu().clone()
                for combo_id, control in self.local_controls.items()
            },
            "counts": dict(self.local_control_counts),
        }

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            self.brats_ddp = False
            self._warn_ddp_once("PEPSY client uses single-process training in this implementation.")

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
                        missing_controls = outputs["missing_control"]
                        combo_ids = outputs["combo_id"]

                        cls_loss = self.loss_fn(logits, y)
                        control_loss = self._control_alignment_loss(combo_ids, missing_controls)
                        contrastive_loss = self._supervised_contrastive_loss(fused_feature, y)
                        loss = (
                            cls_loss
                            + self.control_lambda * control_loss
                            + self.contrastive_lambda * contrastive_loss
                        )

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        control_loss=control_loss.item(),
                        contrastive_loss=contrastive_loss.item(),
                    )
                    pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", ctrl=f"{control_loss.item():.4f}", con=f"{contrastive_loss.item():.4f}")
                    pbar.update(1)

        self.collect_local_controls()
        return self.local_controls
