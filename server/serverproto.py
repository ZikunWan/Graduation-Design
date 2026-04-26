from collections import defaultdict

import torch

from client.clientproto import ClientFedProto
from .serverbase import Server


class ServerFedProto(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_prototypes = None
        self.set_clients(ClientFedProto)

    def get_client_payload(self):
        return {
            "global_prototypes": self.global_prototypes,
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        uploaded_prototypes = [client.get_prototype_payload() for client in active_clients]
        self.global_prototypes = aggregate_prototypes(uploaded_prototypes)
        return self.global_prototypes

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
        last_round_idx = -1
        for round_idx in range(self.global_rounds):
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


def aggregate_prototypes(local_prototype_list):
    aggregated = defaultdict(list)
    for local_prototypes in local_prototype_list:
        for label, prototype in local_prototypes.items():
            aggregated[int(label)].append(prototype.detach().cpu())

    global_prototypes = {}
    for label, prototypes in aggregated.items():
        global_prototypes[int(label)] = torch.stack(prototypes, dim=0).mean(dim=0)
    return global_prototypes
