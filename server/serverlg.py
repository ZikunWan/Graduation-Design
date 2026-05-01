import copy

import torch

from client.clientlg import ClientLGFedAvg
from .serverbase import Server


class ServerLGFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.set_clients(ClientLGFedAvg)
        self.global_classifier_state = state_dict_to_cpu(self.clients[0].get_classifier_state())

    def get_client_payload(self):
        return {
            "classifier_state": self.global_classifier_state,
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        classifier_states = [state_dict_to_cpu(client.get_classifier_state()) for client in active_clients]
        self.global_classifier_state = average_state_dicts(classifier_states, self.uploaded_weights)
        return self.global_classifier_state

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


def average_state_dicts(state_dicts, weights):
    averaged_state = copy.deepcopy(state_dicts[0])
    for key in averaged_state:
        averaged_state[key] = torch.zeros_like(averaged_state[key])

    for state_dict, weight in zip(state_dicts, weights):
        for key in averaged_state:
            averaged_state[key] += state_dict[key] * weight

    return averaged_state
