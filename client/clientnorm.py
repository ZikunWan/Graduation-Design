import torch
from tqdm import tqdm

from .clientbase import Client


class ClientFedNorm(Client):
    def set_parameters(self, payload):
        if payload is None:
            return
        global_state = payload.get("global_model_state")
        if not global_state:
            return

        current_state = self.model.state_dict()
        updated_state = dict(current_state)
        for key, value in global_state.items():
            if key.startswith("modality_norms."):
                continue
            if key in current_state and current_state[key].shape == value.shape:
                updated_state[key] = value.to(current_state[key].device)
        self.model.load_state_dict(updated_state, strict=True)

    def get_model_payload(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def train(self, round_idx, client_idx):
        if self._should_use_brats_ddp():
            self.brats_ddp = False
            self._warn_ddp_once("FedNorm client uses single-process training in this implementation.")

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
                        cls_loss=loss.item(),
                    )
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    pbar.update(1)

        return self.get_model_payload()
