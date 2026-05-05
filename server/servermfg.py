import math

import torch

from client.clientmfg import ClientFedMFG
from dataset import GLOBAL_MODALITIES
from .serverbase import Server


class ServerFedMFG(Server):
    def __init__(self, args):
        super().__init__(args)
        self.mfg_proto_momentum = float(getattr(args, "mfg_proto_momentum"))
        self.mfg_proto_tau = float(getattr(args, "mfg_proto_tau"))
        self.mfg_teacher_lambda = float(getattr(args, "mfg_teacher_lambda"))
        self.mfg_teacher_tau = float(getattr(args, "mfg_teacher_tau"))
        self.mfg_head_eps = float(getattr(args, "mfg_head_eps"))
        self.mfg_head_gamma = float(getattr(args, "mfg_head_gamma"))
        self.mfg_head_tau = float(getattr(args, "mfg_head_tau"))
        self.mfg_head_beta = float(getattr(args, "mfg_head_beta"))
        self.mfg_head_weight_mode = str(getattr(args, "mfg_head_weight_mode"))

        self.global_combo_prototypes = {}
        self.set_clients(ClientFedMFG)
        self.global_classifier_state = state_dict_to_cpu(self.clients[0].get_classifier_state())

    def get_server_state(self):
        return {
            "global_combo_prototypes": self.global_combo_prototypes,
            "global_classifier_state": self.global_classifier_state,
        }

    def load_server_state(self, state):
        if not state:
            self.global_combo_prototypes = {}
            return
        self.global_combo_prototypes = state.get("global_combo_prototypes", {}) or {}
        classifier_state = state.get("global_classifier_state")
        if classifier_state is not None:
            self.global_classifier_state = state_dict_to_cpu(classifier_state)

    def get_client_payload(self):
        return {
            "classifier_state": self.global_classifier_state,
            "teacher_prototypes": build_teacher_prototypes(
                self.global_combo_prototypes,
                teacher_lambda=self.mfg_teacher_lambda,
                teacher_tau=self.mfg_teacher_tau,
            ),
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        payloads = {
            client.client_name: client.get_upload_payload()
            for client in active_clients
        }
        self.global_combo_prototypes = aggregate_combo_prototypes(
            payloads=payloads,
            previous_global=self.global_combo_prototypes,
            momentum=self.mfg_proto_momentum,
            proto_tau=self.mfg_proto_tau,
        )
        self.global_classifier_state = aggregate_classifier_rows(
            payloads=payloads,
            current_global_state=self.global_classifier_state,
            global_combo_prototypes=self.global_combo_prototypes,
            total_modalities=len(GLOBAL_MODALITIES),
            head_tau=self.mfg_head_tau,
            head_beta=self.mfg_head_beta,
            head_weight_mode=self.mfg_head_weight_mode,
        )
        return {
            "global_combo_prototypes": self.global_combo_prototypes,
            "global_classifier_state": self.global_classifier_state,
        }

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


def _normalize_key(key):
    return int(key[0]), int(key[1])


def _combo_size(combo_id):
    return int(combo_id).bit_count()


def _is_subset_combo(base_combo_id, candidate_combo_id):
    base_combo_id = int(base_combo_id)
    candidate_combo_id = int(candidate_combo_id)
    return (base_combo_id & candidate_combo_id) == base_combo_id


def _is_finite_tensor(tensor):
    return torch.is_tensor(tensor) and bool(torch.isfinite(tensor).all())


def _cosine_similarity(left, right):
    left = left.detach().float().view(1, -1)
    right = right.detach().float().view(1, -1)
    return float(torch.nn.functional.cosine_similarity(left, right, dim=1).item())


def aggregate_combo_prototypes(payloads, previous_global, momentum, proto_tau):
    previous_global = {
        _normalize_key(key): value.detach().cpu().clone()
        for key, value in (previous_global or {}).items()
        if _is_finite_tensor(value)
    }
    weighted_sums = {}
    weight_totals = {}
    tau = max(float(proto_tau), 1e-12)

    for payload in payloads.values():
        combo_prototypes = payload.get("combo_prototypes", {})
        combo_counts = payload.get("combo_counts", {})
        for raw_key, prototype in combo_prototypes.items():
            key = _normalize_key(raw_key)
            if not _is_finite_tensor(prototype):
                continue
            count = float(combo_counts.get(raw_key, combo_counts.get(key, 0.0)))
            if count <= 0:
                continue
            weight = count
            previous = previous_global.get(key)
            if previous is not None and _is_finite_tensor(previous):
                weight *= math.exp(_cosine_similarity(prototype, previous) / tau)
            prototype_cpu = prototype.detach().cpu()
            if key not in weighted_sums:
                weighted_sums[key] = torch.zeros_like(prototype_cpu)
                weight_totals[key] = 0.0
            weighted_sums[key] += prototype_cpu * weight
            weight_totals[key] += weight

    candidate_global = {
        key: weighted_sums[key] / weight_totals[key]
        for key in weighted_sums
        if weight_totals[key] > 0
    }

    final_global = {}
    for key, prototype in previous_global.items():
        final_global[key] = prototype.detach().cpu().clone()
    for key, candidate in candidate_global.items():
        previous = previous_global.get(key)
        if previous is not None and _is_finite_tensor(previous):
            final_global[key] = (1.0 - momentum) * previous + momentum * candidate
        else:
            final_global[key] = candidate.detach().cpu().clone()
    return final_global


def build_teacher_prototypes(global_combo_prototypes, teacher_lambda, teacher_tau):
    teacher_lambda = float(teacher_lambda)
    teacher_tau = max(float(teacher_tau), 1e-12)
    normalized = {
        _normalize_key(key): value.detach().cpu().clone()
        for key, value in (global_combo_prototypes or {}).items()
        if _is_finite_tensor(value)
    }
    teachers = {}
    for key, base_prototype in normalized.items():
        combo_id, label = key
        supersets = []
        for other_key, other_prototype in normalized.items():
            other_combo_id, other_label = other_key
            if other_label != label or other_combo_id == combo_id:
                continue
            if _is_subset_combo(combo_id, other_combo_id):
                supersets.append((other_combo_id, other_prototype))

        if not supersets:
            teachers[key] = base_prototype.detach().cpu().clone()
            continue

        weights = torch.tensor(
            [math.exp(_combo_size(other_combo_id) / teacher_tau) for other_combo_id, _ in supersets],
            dtype=base_prototype.dtype,
        )
        stacked = torch.stack([prototype for _, prototype in supersets], dim=0)
        weighted_superset = (stacked * weights.view(-1, 1)).sum(dim=0) / weights.sum()
        teachers[key] = teacher_lambda * base_prototype + (1.0 - teacher_lambda) * weighted_superset
    return teachers


def _client_stats(payload, global_combo_prototypes, total_modalities):
    combo_counts = payload.get("combo_counts", {})
    combo_prototypes = payload.get("combo_prototypes", {})
    total_count = 0.0
    rho_count = 0.0
    rho_numerator = 0.0
    eta_numerator = 0.0

    for raw_key, raw_count in combo_counts.items():
        key = _normalize_key(raw_key)
        combo_id, _ = key
        count = float(raw_count)
        if count <= 0:
            continue
        total_count += count
        eta_numerator += count * (_combo_size(combo_id) / max(float(total_modalities), 1.0))

        local_prototype = combo_prototypes.get(raw_key, combo_prototypes.get(key))
        global_prototype = global_combo_prototypes.get(key)
        if local_prototype is None or global_prototype is None:
            continue
        if not _is_finite_tensor(local_prototype) or not _is_finite_tensor(global_prototype):
            continue
        rho_numerator += count * _cosine_similarity(local_prototype, global_prototype)
        rho_count += count

    if total_count <= 0:
        return 0.0, 0.0, 0.0

    rho = rho_numerator / rho_count if rho_count > 0 else 0.0
    eta = eta_numerator / total_count
    return total_count, rho, eta


def aggregate_classifier_rows(
    payloads,
    current_global_state,
    global_combo_prototypes,
    total_modalities,
    head_tau,
    head_beta,
    head_weight_mode,
):
    aggregated_state = state_dict_to_cpu(current_global_state)
    tau = max(float(head_tau), 1e-12)
    client_weights = []
    client_states = []

    for payload in payloads.values():
        classifier_state = payload.get("classifier_state")
        if classifier_state is None:
            continue
        total_count, rho, eta = _client_stats(
            payload=payload,
            global_combo_prototypes=global_combo_prototypes,
            total_modalities=total_modalities,
        )
        if total_count <= 0:
            continue
        reliability = math.exp(rho / tau)
        if head_weight_mode == "rho_eta":
            reliability *= (1.0 + head_beta * eta)
        client_weights.append(reliability)
        client_states.append(state_dict_to_cpu(classifier_state))

    if not client_states:
        return aggregated_state

    total_weight = sum(client_weights)
    aggregated_state["weight"].zero_()
    aggregated_state["bias"].zero_()
    for client_state, client_weight in zip(client_states, client_weights):
        normalized_weight = client_weight / total_weight
        aggregated_state["weight"] += client_state["weight"] * normalized_weight
        aggregated_state["bias"] += client_state["bias"] * normalized_weight

    return aggregated_state
