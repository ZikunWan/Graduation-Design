import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clientbase import Client
from .clientmm import _move_optimizer_state_to_param_devices


class ClientFedMFG(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.mfg_proto_lambda = float(getattr(args, "mfg_proto_lambda"))
        self.mfg_head_lambda = float(getattr(args, "mfg_head_lambda"))
        self.teacher_prototypes = {}
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

    def _model_module(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def get_classifier_state(self):
        return copy.deepcopy(self._model_module().classifier.state_dict())

    def load_classifier_state(self, classifier_state):
        self._model_module().classifier.load_state_dict(classifier_state, strict=True)

    def set_parameters(self, payload):
        if payload is None:
            self.teacher_prototypes = {}
            return

        classifier_state = payload.get("classifier_state")
        if classifier_state is not None:
            self.load_classifier_state(classifier_state)

        teacher_prototypes = payload.get("teacher_prototypes")
        self.teacher_prototypes = {
            (int(combo_id), int(label)): prototype.detach().cpu().clone()
            for (combo_id, label), prototype in (teacher_prototypes or {}).items()
        }

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

    def _mask_to_combo_id(self, modality_mask_row):
        combo_id = 0
        if torch.is_tensor(modality_mask_row):
            modality_mask_row = modality_mask_row.detach().cpu().tolist()
        for bit_index, value in enumerate(modality_mask_row):
            if value > 0:
                combo_id |= 1 << bit_index
        return int(combo_id)

    def _group_features(self, fused_feature, modality_mask, labels):
        grouped = defaultdict(list)
        labels_list = labels.detach().cpu().tolist()
        modality_mask_list = modality_mask.detach().cpu().tolist()
        for feature, label, mask_row in zip(fused_feature, labels_list, modality_mask_list):
            combo_id = self._mask_to_combo_id(mask_row)
            if combo_id <= 0:
                continue
            if not torch.isfinite(feature).all():
                continue
            grouped[(combo_id, int(label))].append(feature)
        return grouped

    def _batch_prototypes(self, fused_feature, modality_mask, labels):
        grouped = self._group_features(fused_feature, modality_mask, labels)
        prototypes = {}
        for key, features in grouped.items():
            prototypes[key] = torch.stack(features, dim=0).mean(dim=0)
        return prototypes

    def _prototype_alignment_loss(self, batch_prototypes):
        if not self.teacher_prototypes:
            return next(iter(batch_prototypes.values()), None).new_tensor(0.0) if batch_prototypes else self._model_module().classifier.weight.new_tensor(0.0)

        losses = []
        for key, local_prototype in batch_prototypes.items():
            teacher = self.teacher_prototypes.get(key)
            if teacher is None:
                continue
            target = teacher.to(device=local_prototype.device, dtype=local_prototype.dtype)
            if not torch.isfinite(target).all():
                continue
            losses.append(F.mse_loss(local_prototype, target))

        if not losses:
            return self._model_module().classifier.weight.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def _head_calibration_loss(self, batch_prototypes):
        if not self.teacher_prototypes:
            return self._model_module().classifier.weight.new_tensor(0.0)

        logits_list = []
        targets = []
        classifier = self._model_module().classifier
        for key in batch_prototypes:
            teacher = self.teacher_prototypes.get(key)
            if teacher is None:
                continue
            teacher = teacher.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
            if not torch.isfinite(teacher).all():
                continue
            logits_list.append(classifier(teacher.unsqueeze(0)))
            targets.append(int(key[1]))

        if not logits_list:
            return classifier.weight.new_tensor(0.0)

        logits = torch.cat(logits_list, dim=0)
        target = torch.tensor(targets, device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, target)

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
                fused_feature = outputs["fused_feature"].detach().float().cpu()
                modality_mask = outputs["modality_mask"].detach().cpu().tolist()
                labels = y.detach().cpu().tolist()
                for feature, label, mask_row in zip(fused_feature, labels, modality_mask):
                    combo_id = self._mask_to_combo_id(mask_row)
                    if combo_id <= 0:
                        continue
                    if not torch.isfinite(feature).all():
                        continue
                    key = (combo_id, int(label))
                    if key not in grouped_sums:
                        grouped_sums[key] = torch.zeros_like(feature)
                    grouped_sums[key] += feature
                    grouped_counts[key] += 1

        self.local_combo_prototypes = {
            key: feature_sum / grouped_counts[key]
            for key, feature_sum in grouped_sums.items()
            if grouped_counts[key] > 0
        }
        self.local_combo_counts = dict(grouped_counts)
        return self.local_combo_prototypes

    def get_upload_payload(self):
        return {
            "combo_prototypes": {
                key: prototype.detach().cpu().clone()
                for key, prototype in self.local_combo_prototypes.items()
            },
            "combo_counts": dict(self.local_combo_counts),
            "classifier_state": {
                key: value.detach().cpu().clone()
                for key, value in self.get_classifier_state().items()
            },
        }

    def get_prototype_payload(self):
        return self.get_upload_payload()

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            self.brats_ddp = False
            self._warn_ddp_once("FedMFG client uses single-process training in this implementation.")

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
                        modality_mask = outputs["modality_mask"]

                        cls_loss = self.loss_fn(logits, y)
                        batch_prototypes = self._batch_prototypes(fused_feature, modality_mask, y)
                        proto_loss = self._prototype_alignment_loss(batch_prototypes)
                        head_loss = self._head_calibration_loss(batch_prototypes)
                        loss = cls_loss + self.mfg_proto_lambda * proto_loss + self.mfg_head_lambda * head_loss

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        proto_loss=proto_loss.item(),
                        head_loss=head_loss.item(),
                    )
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        cls=f"{cls_loss.item():.4f}",
                        proto=f"{proto_loss.item():.4f}",
                        head=f"{head_loss.item():.4f}",
                    )
                    pbar.update(1)

        self.collect_local_prototypes()
        return self.get_upload_payload()
