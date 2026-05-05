import torch
import torch.nn as nn
from monai.networks.nets import resnet18, resnet34, resnet50

from dataset import GLOBAL_MODALITIES, get_client_spec

RESNET_FACTORY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

BASELINE_ALGOS = {"fedgh", "fedproto", "lgfedavg", "fedtgp", "fd", "local"}
MODALITY_MISSING_BASELINE_ALGOS = {
    "fedamm": "amm",
    "fedmm": "mm",
    "fedmfg": "mfg",
}


def build_resnet_feature_extractor(model_name, spatial_dims, n_input_channels):
    backbone = RESNET_FACTORY[model_name](
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=1,
    )
    feature_extractor = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.act,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    return feature_extractor, backbone.avgpool, backbone.fc.in_features


class MMModel(nn.Module):
    """Per-modality extractor decomposition with local fusion/classifier."""

    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        del prototype_dim
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.is_3d = self.client_spec["is_3d"]
        self.spatial_dims = 3 if self.is_3d else 2
        self.modalities = list(GLOBAL_MODALITIES)
        self.client_modalities = list(self.client_spec["modalities"])

        self.modality_extractors = nn.ModuleDict()
        backbone_dim = None
        for modality in self.client_modalities:
            feature_extractor, global_pool, feature_dim = build_resnet_feature_extractor(
                model_name,
                self.spatial_dims,
                n_input_channels=1,
            )
            self.modality_extractors[modality] = feature_extractor
            if backbone_dim is None:
                backbone_dim = feature_dim
                self.global_pool = global_pool

        self.backbone_dim = backbone_dim
        self.fused_dim = backbone_dim * len(self.client_modalities)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        self.model_parallel = False
        self.output_device = None
        self.modality_devices = {}

    def configure_model_parallel(self, devices):
        devices = [torch.device(device) for device in devices]
        if not devices:
            return

        self.model_parallel = len(devices) > 1
        self.output_device = devices[0]
        self.modality_devices = {}
        for index, modality in enumerate(self.client_modalities):
            device = devices[index % len(devices)]
            self.modality_devices[modality] = device
            self.modality_extractors[modality].to(device)

        self.classifier.to(self.output_device)
        self.dropout.to(self.output_device)

    def _runtime_dtype(self, device):
        if device.type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return self.classifier.weight.dtype

    def _encode_one_modality(self, modality, tensor):
        target_device = self.modality_devices.get(modality)
        if target_device is not None and tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=True)
        feature_map = self.modality_extractors[modality](tensor)
        feature_map = self.global_pool(feature_map)
        return torch.flatten(feature_map, 1)

    def extract_modality_features(self, x):
        modality_mask = x["modality_mask"]
        batch_size = modality_mask.shape[0]
        feature_dtype = self._runtime_dtype(modality_mask.device)
        features = []

        for modality_index, modality in enumerate(self.modalities):
            feature = torch.zeros(
                batch_size,
                self.backbone_dim,
                device=modality_mask.device,
                dtype=feature_dtype,
            )
            if modality in self.modality_extractors and modality in x["modalities"]:
                active_index = modality_mask[:, modality_index] > 0
                if active_index.any():
                    modality_tensor = x["modalities"][modality]
                    tensor_active_index = active_index
                    if tensor_active_index.device != modality_tensor.device:
                        tensor_active_index = tensor_active_index.to(modality_tensor.device)
                    encoded = self._encode_one_modality(modality, modality_tensor[tensor_active_index])
                    feature[active_index] = encoded.to(device=feature.device, dtype=feature.dtype)
            features.append(feature)

        return torch.stack(features, dim=1), modality_mask

    def extract_modality_prototypes(self, modality_features, modality_mask):
        del modality_mask
        return modality_features

    def encode(self, x):
        modality_features, modality_mask = self.extract_modality_features(x)
        selected_features = []
        for modality in self.client_modalities:
            modality_index = self.modalities.index(modality)
            selected_features.append(modality_features[:, modality_index, :])
        fused_feature = torch.cat(selected_features, dim=1)
        fused_feature = self.dropout(fused_feature)
        return {
            "modality_features": modality_features,
            "modality_mask": modality_mask,
            "fused_feature": fused_feature,
        }

    def forward(self, x, return_prototype=False, return_feature=False, return_dict=False):
        encoded = self.encode(x)
        fused_feature = encoded["fused_feature"]
        logits = self.classifier(fused_feature)

        if return_dict:
            output = {
                "logits": logits,
                **encoded,
            }
            if return_prototype:
                output["modality_prototypes"] = self.extract_modality_prototypes(
                    encoded["modality_features"],
                    encoded["modality_mask"],
                )
            return output

        if return_prototype and return_feature:
            modality_prototypes = self.extract_modality_prototypes(
                encoded["modality_features"],
                encoded["modality_mask"],
            )
            return logits, fused_feature, modality_prototypes, encoded["modality_mask"]

        if return_prototype:
            modality_prototypes = self.extract_modality_prototypes(
                encoded["modality_features"],
                encoded["modality_mask"],
            )
            return logits, modality_prototypes, encoded["modality_mask"]

        if return_feature:
            return logits, fused_feature

        return logits


class AMMModel(MMModel):
    """FedAMM model with a projection head for prototype-level modality balance."""

    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__(
            client_name=client_name,
            num_classes=num_classes,
            model_name=model_name,
            prototype_dim=prototype_dim,
            dropout=dropout,
        )
        self.amm_unimodal_projection = self._build_amm_projection_head(
            input_dim=self.backbone_dim,
            output_dim=self.fused_dim,
            hidden_dim=prototype_dim,
        )

    def _build_amm_projection_head(self, input_dim, output_dim, hidden_dim):
        if input_dim == output_dim:
            return nn.Identity()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            return nn.Linear(input_dim, output_dim)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def configure_model_parallel(self, devices):
        super().configure_model_parallel(devices)
        if self.output_device is not None:
            self.amm_unimodal_projection.to(self.output_device)

    def project_amm_unimodal_embeddings(self, modality_features):
        return self.amm_unimodal_projection(modality_features)


class MFGModel(MMModel):
    """Modality-aware projected fusion with a shared classifier head."""

    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__(
            client_name=client_name,
            num_classes=num_classes,
            model_name=model_name,
            prototype_dim=prototype_dim,
            dropout=dropout,
        )
        embedding_dim = int(prototype_dim)
        if embedding_dim <= 0:
            embedding_dim = self.backbone_dim
        self.embedding_dim = embedding_dim
        self.modality_projectors = nn.ModuleDict(
            {
                modality: nn.Linear(self.backbone_dim, self.embedding_dim)
                for modality in self.client_modalities
            }
        )
        self.modality_gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embedding_dim, 1),
        )
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def configure_model_parallel(self, devices):
        super().configure_model_parallel(devices)
        if self.output_device is None:
            return
        for projector in self.modality_projectors.values():
            projector.to(self.output_device)
        self.modality_gate.to(self.output_device)
        self.classifier.to(self.output_device)

    def _project_modality_features(self, modality_features):
        batch_size = modality_features.shape[0]
        projected = torch.zeros(
            batch_size,
            len(self.modalities),
            self.embedding_dim,
            device=modality_features.device,
            dtype=modality_features.dtype,
        )
        for modality in self.client_modalities:
            modality_index = self.modalities.index(modality)
            projected[:, modality_index, :] = self.modality_projectors[modality](
                modality_features[:, modality_index, :]
            )
        return projected

    def encode(self, x):
        modality_features, modality_mask = self.extract_modality_features(x)
        projected_features = self._project_modality_features(modality_features)
        gate_logits = self.modality_gate(projected_features).squeeze(-1)
        active_mask = modality_mask > 0
        gate_floor = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill(~active_mask, gate_floor)
        attention = torch.softmax(gate_logits, dim=1)
        attention = attention * active_mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        fused_feature = torch.sum(projected_features * attention.unsqueeze(-1), dim=1)
        fused_feature = self.dropout(fused_feature)
        return {
            "modality_features": projected_features,
            "modality_mask": modality_mask,
            "fused_feature": fused_feature,
            "attention": attention,
        }


class BaselineBrainTumorModel(nn.Module):
    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        del prototype_dim
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.is_3d = self.client_spec["is_3d"]
        self.spatial_dims = 3 if self.is_3d else 2
        self.modalities = list(self.client_spec["modalities"])

        backbone = RESNET_FACTORY[model_name](
            spatial_dims=self.spatial_dims,
            n_input_channels=len(self.modalities),
            num_classes=1,
        )
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.act,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.global_pool = backbone.avgpool
        self.backbone_dim = backbone.fc.in_features
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone_dim, num_classes)

    def _stack_modalities(self, x):
        ordered_modalities = x.get("modality_order", self.modalities)
        stacked_inputs = [x["modalities"][modality] for modality in ordered_modalities]
        return torch.cat(stacked_inputs, dim=1)

    def encode(self, x):
        stacked_input = self._stack_modalities(x)
        feature_map = self.feature_extractor(stacked_input)
        feature_map = self.global_pool(feature_map)
        fused_feature = torch.flatten(feature_map, 1)
        fused_feature = self.dropout(fused_feature)
        return fused_feature

    def forward(self, x, return_prototype=False, return_feature=False, return_dict=False):
        if return_prototype:
            raise NotImplementedError(
                "BaselineBrainTumorModel does not provide modality-level prototypes. "
                "Use model_mode='multimodal' for method-specific prototype logic."
            )

        fused_feature = self.encode(x)
        logits = self.classifier(fused_feature)

        if return_dict:
            return {
                "logits": logits,
                "fused_feature": fused_feature,
            }

        if return_feature:
            return logits, fused_feature

        return logits


def resolve_model_mode(model_mode, algo):
    if algo is not None:
        algo = str(algo).lower()
    if model_mode != "auto":
        if model_mode == "multimodal":
            return MODALITY_MISSING_BASELINE_ALGOS.get(algo, "amm")
        if model_mode == "amm":
            return "amm"
        return model_mode
    if algo in BASELINE_ALGOS:
        return "baseline"
    return MODALITY_MISSING_BASELINE_ALGOS.get(algo, "amm")


def build_client_model(
    client_name,
    num_classes=5,
    model_name="resnet18",
    prototype_dim=256,
    dropout=0.0,
    model_mode="auto",
    algo=None,
):
    resolved_mode = resolve_model_mode(model_mode, algo)
    model_classes = {
        "baseline": BaselineBrainTumorModel,
        "amm": AMMModel,
        "mm": MMModel,
        "mfg": MFGModel,
    }
    if resolved_mode not in model_classes:
        raise ValueError(
            f"Unknown model mode '{resolved_mode}'. "
            f"Expected one of {sorted(model_classes)}."
        )
    model_cls = model_classes[resolved_mode]
    model = model_cls(
        client_name=client_name,
        num_classes=num_classes,
        model_name=model_name,
        prototype_dim=prototype_dim,
        dropout=dropout,
    )
    model.model_mode = resolved_mode
    return model
