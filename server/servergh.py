import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from client.clientbase import move_optimizer_state
from client.clientgh import ClientFedGH
from .serverbase import Server


class ServerFedGH(Server):
    def __init__(self, args):
        super().__init__(args)
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs

        self.set_clients(ClientFedGH)
        self.global_head = copy.deepcopy(self.clients[0].model.classifier).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.server_optimizer = torch.optim.AdamW(self.global_head.parameters(), lr=self.server_learning_rate)

    def get_server_state(self):
        return {
            "global_head": self.global_head.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
        }

    def load_server_state(self, state):
        if "global_head" in state:
            self.global_head.load_state_dict(state["global_head"])
        if "server_optimizer" in state:
            self.server_optimizer.load_state_dict(state["server_optimizer"])
            move_optimizer_state(self.server_optimizer, torch.device(self.device))

    def get_client_payload(self):
        return {
            "classifier_state": state_dict_to_cpu(self.global_head.state_dict()),
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        uploaded_prototypes = []
        for client in active_clients:
            local_prototypes = client.get_prototype_payload()
            for label, prototype in local_prototypes.items():
                uploaded_prototypes.append((prototype.detach().cpu(), int(label)))

        if not uploaded_prototypes:
            return state_dict_to_cpu(self.global_head.state_dict())

        proto_loader = DataLoader(uploaded_prototypes, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        self.global_head.train()
        for _ in range(self.server_epochs):
            for proto_batch, label_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                logits = self.global_head(proto_batch)
                loss = self.loss_fn(logits, label_batch)

                self.server_optimizer.zero_grad()
                loss.backward()
                self.server_optimizer.step()

        return state_dict_to_cpu(self.global_head.state_dict())

    def train_round(self, round_idx):
        self.selected_clients = self.select_clients()
        self.send_parameters()

        self.train_clients(self.selected_clients, round_idx)

        active_clients = self.receive_ids()
        self.aggregate(active_clients)
        self.send_parameters()

        elapsed_time = self._record_round_completion(round_idx)
        self.maybe_evaluate_validation(round_idx, elapsed_time)
        self.maybe_save_checkpoint(round_idx)

    def train(self):
        self._start_training_timer()
        start_round = self.get_resume_start_round()
        last_round_idx = self.get_last_completed_round()
        for round_idx in range(start_round, self.global_rounds):
            last_round_idx = round_idx
            self.train_round(round_idx)
            if self.should_stop_training():
                break

        final_round_idx = self.restore_best_state_for_final_test(last_round_idx if last_round_idx >= 0 else 0)
        final_summary = self.evaluate(split="test", round_idx=final_round_idx)
        final_summary["round"] = final_round_idx
        final_summary["elapsed_time"] = self._elapsed_since_training_start()
        self._print_round_summary(final_summary)
        self.save_checkpoint(final_round_idx)
        return [final_summary]


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }
