from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clientbase import Client


class ClientFedMM(Client):
    def __init__(self, args, client_name, **kwargs):
        super().__init__(args, client_name, **kwargs)
        self.proto_lambda = float(getattr(args, "fedmm_proto_lambda", getattr(args, "proto_lambda", 1.0)))
        self.global_modality_prototypes = {}
        self.local_modality_prototypes = {}
        self.local_modality_counts = {}

    def set_parameters(self, payload):
        if payload is None:
            self.global_modality_prototypes = {}
            return
        prototypes = payload.get("global_modality_prototypes")
        self.global_modality_prototypes = {
            (str(modality), int(label)): prototype.detach().cpu().clone()
            for (modality, label), prototype in (prototypes or {}).items()
        }

    def _prototype_alignment_loss(self, modality_prototypes, modality_mask, labels):
        if not self.global_modality_prototypes:
            return modality_prototypes.new_tensor(0.0)

        losses = []
        for modality_index, modality in enumerate(self.model.modalities):
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
                losses.append(1.0 - F.cosine_similarity(
                    local_prototype.unsqueeze(0),
                    global_prototype.to(self.device).unsqueeze(0),
                    dim=1,
                ).mean())

        if not losses:
            return modality_prototypes.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def collect_local_prototypes(self):
        trainloader = self.load_train_data(shuffle=False)
        self.model.eval()
        grouped = defaultdict(list)
        counts = defaultdict(int)

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
                    for modality_index, modality in enumerate(self.model.modalities):
                        if modality_mask[row_index, modality_index] <= 0:
                            continue
                        key = (modality, int(label))
                        grouped[key].append(modality_prototypes[row_index, modality_index])
                        counts[key] += 1

        self.local_modality_prototypes = {
            key: F.normalize(torch.stack(features, dim=0).mean(dim=0), dim=0)
            for key, features in grouped.items()
        }
        self.local_modality_counts = dict(counts)
        return self.local_modality_prototypes

    def get_prototype_payload(self):
        return {
            "modality_prototypes": {
                key: prototype.detach().cpu().clone()
                for key, prototype in self.local_modality_prototypes.items()
            },
            "modality_counts": dict(self.local_modality_counts),
        }

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            self.brats_ddp = False
            self._warn_ddp_once("FedMM client uses single-process training in this implementation.")

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
                        proto_loss = self._prototype_alignment_loss(modality_prototypes, modality_mask, y)
                        loss = cls_loss + self.proto_lambda * proto_loss

                    self.backward_step(loss)
                    self.record_train_step(
                        round_idx=round_idx,
                        local_epoch=local_epoch,
                        loss=loss.item(),
                        cls_loss=cls_loss.item(),
                        proto_loss=proto_loss.item(),
                    )
                    pbar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", proto=f"{proto_loss.item():.4f}")
                    pbar.update(1)

        self.collect_local_prototypes()
        return self.local_modality_prototypes
