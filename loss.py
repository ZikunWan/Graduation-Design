import torch
import torch.nn.functional as F


def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def prototype_alignment_loss(
    local_prototype,
    local_prototype_mask,
    global_prototype,
    global_prototype_mask,
    local_prototype_count=None,
):
    global_prototype = global_prototype.unsqueeze(2)
    global_prototype_mask = global_prototype_mask.unsqueeze(2)

    cosine_distance = 1.0 - F.cosine_similarity(local_prototype, global_prototype, dim=-1)
    valid_mask = local_prototype_mask * global_prototype_mask

    if local_prototype_count is None:
        weight = valid_mask
    else:
        weight = valid_mask * local_prototype_count

    loss = (cosine_distance * weight).sum() / weight.sum()
    return loss


def total_loss(
    logits,
    labels,
    local_prototype,
    local_prototype_mask,
    global_prototype,
    global_prototype_mask,
    local_prototype_count=None,
    lambda_proto=1.0,
):
    cls_loss = classification_loss(logits, labels)
    proto_loss = prototype_alignment_loss(
        local_prototype,
        local_prototype_mask,
        global_prototype,
        global_prototype_mask,
        local_prototype_count=local_prototype_count,
    )
    loss = cls_loss + lambda_proto * proto_loss
    return loss, cls_loss, proto_loss
