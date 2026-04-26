from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from client.clienttgp import ClientFedTGP
from .serverbase import Server


class TrainableGlobalPrototypes(nn.Module):
    def __init__(self, num_classes, feature_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, feature_dim)
        self.hidden = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, class_ids):
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=self.embedding.weight.device)
        embedded = self.embedding(class_ids)
        hidden = self.hidden(embedded)
        return self.output(hidden)


class ServerFedTGP(Server):
    def __init__(self, args):
        super().__init__(args)
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs
        self.margin_threshold = args.margin_threshold

        self.set_clients(ClientFedTGP)
        self.feature_dim = self.clients[0].model.backbone_dim
        self.tgp = TrainableGlobalPrototypes(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
        ).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.server_optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.server_learning_rate)
        self.global_prototypes = self._export_global_prototypes()

    def get_server_state(self):
        return {
            "tgp": self.tgp.state_dict(),
            "server_optimizer": self.server_optimizer.state_dict(),
        }

    def load_server_state(self, state):
        if "tgp" in state:
            self.tgp.load_state_dict(state["tgp"])
        if "server_optimizer" in state:
            self.server_optimizer.load_state_dict(state["server_optimizer"])

    def _export_global_prototypes(self):
        self.tgp.eval()
        global_prototypes = {}
        with torch.no_grad():
            class_ids = torch.arange(self.num_classes, device=self.device)
            prototypes = self.tgp(class_ids).detach().cpu()
            for class_id in range(self.num_classes):
                global_prototypes[class_id] = prototypes[class_id]
        return global_prototypes

    def get_client_payload(self):
        return {
            "global_prototypes": self.global_prototypes,
        }

    def send_parameters(self):
        payload = self.get_client_payload()
        for client in self.selected_clients:
            client.set_parameters(payload)

    def aggregate(self, active_clients):
        uploaded_prototypes = []
        classwise_prototypes = defaultdict(list)
        for client in active_clients:
            local_prototypes = client.get_prototype_payload()
            for label, prototype in local_prototypes.items():
                cpu_proto = prototype.detach().cpu()
                uploaded_prototypes.append((cpu_proto, int(label)))
                classwise_prototypes[int(label)].append(cpu_proto)

        if not uploaded_prototypes:
            return self.global_prototypes

        margin = compute_margin(classwise_prototypes, self.margin_threshold)
        proto_loader = DataLoader(uploaded_prototypes, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        self.tgp.train()
        for _ in range(self.server_epochs):
            for proto_batch, label_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                class_ids = torch.arange(self.num_classes, device=self.device)
                generated_prototypes = self.tgp(class_ids)
                distances = torch.cdist(proto_batch, generated_prototypes, p=2)
                one_hot = F.one_hot(label_batch, self.num_classes).float()
                logits = -(distances + one_hot * margin)
                loss = self.loss_fn(logits, label_batch)

                self.server_optimizer.zero_grad()
                loss.backward()
                self.server_optimizer.step()

        self.global_prototypes = self._export_global_prototypes()
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


def compute_margin(classwise_prototypes, margin_threshold):
    labels = sorted(classwise_prototypes.keys())
    if len(labels) < 2:
        return 0.0

    averaged = {}
    for label, prototypes in classwise_prototypes.items():
        averaged[label] = torch.stack(prototypes, dim=0).mean(dim=0)

    min_distances = []
    for index, label_a in enumerate(labels):
        distances = []
        for label_b in labels:
            if label_a == label_b:
                continue
            distances.append(torch.norm(averaged[label_a] - averaged[label_b], p=2).item())
        if distances:
            min_distances.append(min(distances))

    if not min_distances:
        return 0.0
    return min(max(min_distances), margin_threshold)
