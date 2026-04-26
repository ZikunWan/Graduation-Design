import torch

from client.clientnorm import ClientFedNorm
from .serverbase import Server


class ServerFedNorm(Server):
    def __init__(self, args):
        super().__init__(args)
        self.global_model_state = None
        self.set_clients(ClientFedNorm)

    def get_server_state(self):
        return {"global_model_state": self.global_model_state}

    def load_server_state(self, state):
        self.global_model_state = state.get("global_model_state") if state else None

    def get_client_payload(self):
        return {"global_model_state": self.global_model_state}

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        payloads = [client.get_model_payload() for client in active_clients]
        weights = [client.train_samples for client in active_clients]
        self.global_model_state = aggregate_compatible_non_norm_state(payloads, weights)
        return self.global_model_state

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


def _is_local_norm_key(key):
    return (
        key.startswith("modality_norms.")
        or key.endswith("running_mean")
        or key.endswith("running_var")
        or key.endswith("num_batches_tracked")
    )


def aggregate_compatible_non_norm_state(payloads, weights):
    if not payloads:
        return None

    total_weight = float(sum(weights)) if sum(weights) > 0 else float(len(payloads))
    normalized_weights = [float(weight) / total_weight for weight in weights]
    global_state = {}

    candidate_keys = set().union(*(payload.keys() for payload in payloads))
    for key in sorted(candidate_keys):
        if _is_local_norm_key(key):
            continue

        values = [payload.get(key) for payload in payloads]
        if any(value is None for value in values):
            continue
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) != 1:
            continue
        if not torch.is_floating_point(values[0]):
            continue

        stacked = torch.stack(
            [value.detach().cpu() * normalized_weights[index] for index, value in enumerate(values)],
            dim=0,
        )
        global_state[key] = stacked.sum(dim=0)

    return global_state
