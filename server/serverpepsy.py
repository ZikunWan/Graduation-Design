from collections import defaultdict

import torch

from client.clientpepsy import ClientPEPSY
from .serverbase import Server


class ServerPEPSY(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_controls = {}
        self.set_clients(ClientPEPSY)

    def get_server_state(self):
        return {"global_controls": self.global_controls}

    def load_server_state(self, state):
        self.global_controls = state.get("global_controls", {}) if state else {}

    def get_client_payload(self):
        return {"global_controls": self.global_controls}

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        payloads = [client.get_control_payload() for client in active_clients]
        self.global_controls = aggregate_controls(payloads)
        return self.global_controls

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


def aggregate_controls(payloads):
    weighted_controls = defaultdict(list)
    weights = defaultdict(list)
    for payload in payloads:
        controls = payload.get("controls", {})
        counts = payload.get("counts", {})
        for combo_id, control in controls.items():
            combo_id = int(combo_id)
            count = float(counts.get(combo_id, 1.0))
            weighted_controls[combo_id].append(control.detach().cpu() * count)
            weights[combo_id].append(count)

    global_controls = {}
    for combo_id, controls in weighted_controls.items():
        total_weight = max(sum(weights[combo_id]), 1.0)
        global_controls[int(combo_id)] = torch.stack(controls, dim=0).sum(dim=0) / total_weight
    return global_controls
