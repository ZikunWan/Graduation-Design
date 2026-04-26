import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18, resnet34, resnet50

from dataset import GLOBAL_MODALITIES, get_client_spec

RESNET_FACTORY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}

BASELINE_ALGOS = {"fedgh", "fedproto", "lgfedavg", "fedtgp", "fd", "local"}
METHOD_MODEL_MODES = {
    "pepsy": "pepsy",
    "fedamm": "amm",
    "amm": "amm",
    "fedmm": "mm",
    "mm": "mm",
    "fednorm": "norm",
    "norm": "norm",
}


def build_resnet_front(model_name, spatial_dims):
    backbone = RESNET_FACTORY[model_name](
        spatial_dims=spatial_dims,
        n_input_channels=1,
        num_classes=1,
    )
    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.act,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
    )


def build_resnet_back(model_name, spatial_dims):
    backbone = RESNET_FACTORY[model_name](
        spatial_dims=spatial_dims,
        n_input_channels=1,
        num_classes=1,
    )
    shared_back = nn.Sequential(
        backbone.layer3,
        backbone.layer4,
    )
    return shared_back, backbone.avgpool, backbone.fc.in_features


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


class MaskedAttentionFusion(nn.Module):
    def __init__(self, feature_dim, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, modality_features, modality_mask):
        scores = self.score(modality_features).squeeze(-1)
        scores = scores.masked_fill(modality_mask <= 0, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1) * modality_mask
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused_feature = (modality_features * attention.unsqueeze(-1)).sum(dim=1)
        return fused_feature, attention


class PEPSYModel(nn.Module):
    """Missing-pattern controls plus representation reconfiguration."""

    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.is_3d = self.client_spec["is_3d"]
        self.spatial_dims = 3 if self.is_3d else 2
        self.modalities = list(self.client_spec["modalities"])
        self.full_modalities = list(GLOBAL_MODALITIES)
        self.prototype_dim = prototype_dim

        self.feature_extractor, self.global_pool, backbone_dim = build_resnet_feature_extractor(
            model_name,
            self.spatial_dims,
            n_input_channels=len(self.modalities),
        )
        self.backbone_dim = backbone_dim
        self.control_dim = prototype_dim
        self.num_patterns = 1 << len(self.full_modalities)
        self.pattern_controls = nn.Embedding(self.num_patterns, self.control_dim)
        self.reconfiguration = nn.Sequential(
            nn.Linear(backbone_dim + self.control_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, backbone_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_dim, num_classes)
        self.prototype_head = nn.Sequential(
            nn.Linear(backbone_dim, prototype_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "combo_powers",
            torch.tensor([1 << index for index in range(len(self.full_modalities))], dtype=torch.long),
            persistent=False,
        )

    def _stack_modalities(self, x):
        ordered_modalities = x.get("modality_order", self.modalities)
        stacked_inputs = [x["modalities"][modality] for modality in ordered_modalities]
        return torch.cat(stacked_inputs, dim=1)

    def _mask_to_combo_id(self, modality_mask):
        powers = self.combo_powers.to(modality_mask.device)
        return ((modality_mask > 0).long() * powers.unsqueeze(0)).sum(dim=1)

    def encode(self, x):
        stacked_input = self._stack_modalities(x)
        feature_map = self.feature_extractor(stacked_input)
        feature_map = self.global_pool(feature_map)
        raw_feature = torch.flatten(feature_map, 1)

        modality_mask = x["modality_mask"]
        combo_id = self._mask_to_combo_id(modality_mask)
        missing_control = self.pattern_controls(combo_id)
        delta = self.reconfiguration(torch.cat([raw_feature, missing_control], dim=1))
        fused_feature = self.dropout(raw_feature + delta)

        return {
            "raw_feature": raw_feature,
            "fused_feature": fused_feature,
            "missing_control": missing_control,
            "combo_id": combo_id,
            "modality_mask": modality_mask,
        }

    def extract_prototypes(self, fused_feature):
        return F.normalize(self.prototype_head(fused_feature), dim=1)

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
                output["prototypes"] = self.extract_prototypes(fused_feature)
            return output

        if return_prototype and return_feature:
            return logits, fused_feature, self.extract_prototypes(fused_feature), encoded["modality_mask"]

        if return_prototype:
            return logits, self.extract_prototypes(fused_feature), encoded["modality_mask"]

        if return_feature:
            return logits, fused_feature

        return logits


class AMMModel(nn.Module):
    """Modality-specific encoders, multimodal fusion, and prototype outputs."""

    def __init__(
        self,
        client_name,
        num_classes=5,
        model_name="resnet18",
        prototype_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.is_3d = self.client_spec["is_3d"]
        self.spatial_dims = 3 if self.is_3d else 2
        self.modalities = list(GLOBAL_MODALITIES)
        self.prototype_dim = prototype_dim

        self.modality_extractors = nn.ModuleDict()
        backbone_dim = None
        for modality in self.modalities:
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
        self.fusion = MaskedAttentionFusion(backbone_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_dim, num_classes)
        self.modality_prototype_heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(backbone_dim, prototype_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for modality in self.modalities
            }
        )

    def _runtime_dtype(self, device):
        if device.type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return self.classifier.weight.dtype

    def _encode_one_modality(self, modality, tensor):
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
            if modality in x["modalities"]:
                active_index = modality_mask[:, modality_index] > 0
                if active_index.any():
                    encoded = self._encode_one_modality(modality, x["modalities"][modality][active_index])
                    feature[active_index] = encoded.to(feature.dtype)
            features.append(feature)

        return torch.stack(features, dim=1), modality_mask

    def fuse_features(self, modality_features, modality_mask):
        fused_feature, attention = self.fusion(modality_features, modality_mask)
        return self.dropout(fused_feature), attention

    def extract_modality_prototypes(self, modality_features, modality_mask):
        batch_size = modality_mask.shape[0]
        prototype_dtype = self._runtime_dtype(modality_features.device)
        prototypes = []

        for modality_index, modality in enumerate(self.modalities):
            prototype = torch.zeros(
                batch_size,
                self.prototype_dim,
                device=modality_features.device,
                dtype=prototype_dtype,
            )
            active_index = modality_mask[:, modality_index] > 0
            if active_index.any():
                active_feature = modality_features[active_index, modality_index]
                active_prototype = self.modality_prototype_heads[modality](active_feature)
                prototype[active_index] = F.normalize(active_prototype, dim=1).to(prototype.dtype)
            prototypes.append(prototype)

        return torch.stack(prototypes, dim=1)

    def encode(self, x):
        modality_features, modality_mask = self.extract_modality_features(x)
        fused_feature, fusion_attention = self.fuse_features(modality_features, modality_mask)
        return {
            "modality_features": modality_features,
            "modality_mask": modality_mask,
            "fusion_attention": fusion_attention,
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
        self.client_name = client_name
        self.client_spec = get_client_spec(client_name)
        self.is_3d = self.client_spec["is_3d"]
        self.spatial_dims = 3 if self.is_3d else 2
        self.modalities = list(GLOBAL_MODALITIES)
        self.client_modalities = list(self.client_spec["modalities"])
        self.prototype_dim = prototype_dim

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
        self.fusion = MaskedAttentionFusion(backbone_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_dim, num_classes)
        self.modality_prototype_heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(backbone_dim, prototype_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for modality in self.client_modalities
            }
        )

    def _runtime_dtype(self, device):
        if device.type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_dtype("cuda")
        return self.classifier.weight.dtype

    def _encode_one_modality(self, modality, tensor):
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
                    encoded = self._encode_one_modality(modality, x["modalities"][modality][active_index])
                    feature[active_index] = encoded.to(feature.dtype)
            features.append(feature)

        return torch.stack(features, dim=1), modality_mask

    def extract_modality_prototypes(self, modality_features, modality_mask):
        batch_size = modality_mask.shape[0]
        prototype_dtype = self._runtime_dtype(modality_features.device)
        prototypes = []

        for modality_index, modality in enumerate(self.modalities):
            prototype = torch.zeros(
                batch_size,
                self.prototype_dim,
                device=modality_features.device,
                dtype=prototype_dtype,
            )
            if modality in self.modality_prototype_heads:
                active_index = modality_mask[:, modality_index] > 0
                if active_index.any():
                    active_feature = modality_features[active_index, modality_index]
                    active_prototype = self.modality_prototype_heads[modality](active_feature)
                    prototype[active_index] = F.normalize(active_prototype, dim=1).to(prototype.dtype)
            prototypes.append(prototype)

        return torch.stack(prototypes, dim=1)

    def encode(self, x):
        modality_features, modality_mask = self.extract_modality_features(x)
        fused_feature, fusion_attention = self.fusion(modality_features, modality_mask)
        fused_feature = self.dropout(fused_feature)
        return {
            "modality_features": modality_features,
            "modality_mask": modality_mask,
            "fusion_attention": fusion_attention,
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


class NormModel(nn.Module):
    """Stacked-modality classifier with modality-specific local normalization."""

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

        norm_cls = nn.BatchNorm3d if self.is_3d else nn.BatchNorm2d
        self.modality_norms = nn.ModuleDict(
            {
                modality: norm_cls(1, affine=True, track_running_stats=True)
                for modality in self.modalities
            }
        )
        self.feature_extractor, self.global_pool, backbone_dim = build_resnet_feature_extractor(
            model_name,
            self.spatial_dims,
            n_input_channels=len(self.modalities),
        )
        self.backbone_dim = backbone_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def _stack_normalized_modalities(self, x):
        ordered_modalities = x.get("modality_order", self.modalities)
        normalized_inputs = []
        for modality in ordered_modalities:
            tensor = x["modalities"][modality]
            if modality in self.modality_norms:
                tensor = self.modality_norms[modality](tensor)
            normalized_inputs.append(tensor)
        return torch.cat(normalized_inputs, dim=1)

    def encode(self, x):
        stacked_input = self._stack_normalized_modalities(x)
        feature_map = self.feature_extractor(stacked_input)
        feature_map = self.global_pool(feature_map)
        fused_feature = torch.flatten(feature_map, 1)
        fused_feature = self.dropout(fused_feature)
        return {
            "fused_feature": fused_feature,
            "modality_mask": x["modality_mask"],
        }

    def forward(self, x, return_prototype=False, return_feature=False, return_dict=False):
        if return_prototype:
            raise NotImplementedError("NormModel does not provide modality-level prototypes.")

        encoded = self.encode(x)
        fused_feature = encoded["fused_feature"]
        logits = self.classifier(fused_feature)

        if return_dict:
            return {
                "logits": logits,
                **encoded,
            }

        if return_feature:
            return logits, fused_feature

        return logits


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
            return METHOD_MODEL_MODES.get(algo, "amm")
        return METHOD_MODEL_MODES.get(str(model_mode).lower(), model_mode)
    if algo in BASELINE_ALGOS:
        return "baseline"
    return METHOD_MODEL_MODES.get(algo, "amm")


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
        "pepsy": PEPSYModel,
        "amm": AMMModel,
        "mm": MMModel,
        "norm": NormModel,
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
