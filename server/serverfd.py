from collections import defaultdict

import torch

from client.clientfd import ClientFD
from .serverbase import Server


class ServerFD(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_logits = None
        self.set_clients(ClientFD)

    def get_client_payload(self):
        return {
            "global_logits": self.global_logits,
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        uploaded_logits = [client.get_logits_payload() for client in active_clients]
        self.global_logits = aggregate_logits(uploaded_logits)
        return self.global_logits

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


def aggregate_logits(local_logits_list):
    aggregated_logits = defaultdict(list)
    for local_logits in local_logits_list:
        for label, logit in local_logits.items():
            aggregated_logits[int(label)].append(logit.detach().cpu())

    global_logits = {}
    for label, logits_list in aggregated_logits.items():
        global_logits[int(label)] = torch.stack(logits_list, dim=0).mean(dim=0)
    return global_logits
