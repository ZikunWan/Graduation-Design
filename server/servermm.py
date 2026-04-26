from collections import defaultdict

import torch

from client.clientmm import ClientFedMM
from .serverbase import Server


class ServerFedMM(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_modality_prototypes = {}
        self.set_clients(ClientFedMM)

    def get_server_state(self):
        return {"global_modality_prototypes": self.global_modality_prototypes}

    def load_server_state(self, state):
        self.global_modality_prototypes = state.get("global_modality_prototypes", {}) if state else {}

    def get_client_payload(self):
        return {"global_modality_prototypes": self.global_modality_prototypes}

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        payloads = [client.get_prototype_payload() for client in active_clients]
        self.global_modality_prototypes = aggregate_modality_prototypes(payloads)
        return self.global_modality_prototypes

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


def aggregate_modality_prototypes(payloads):
    weighted = defaultdict(list)
    weights = defaultdict(list)
    for payload in payloads:
        prototypes = payload.get("modality_prototypes", {})
        counts = payload.get("modality_counts", {})
        for key, prototype in prototypes.items():
            modality, label = str(key[0]), int(key[1])
            count = float(counts.get(key, 1.0))
            weighted[(modality, label)].append(prototype.detach().cpu() * count)
            weights[(modality, label)].append(count)

    global_prototypes = {}
    for key, prototypes in weighted.items():
        total_weight = max(sum(weights[key]), 1.0)
        prototype = torch.stack(prototypes, dim=0).sum(dim=0) / total_weight
        global_prototypes[key] = torch.nn.functional.normalize(prototype, dim=0)
    return global_prototypes
