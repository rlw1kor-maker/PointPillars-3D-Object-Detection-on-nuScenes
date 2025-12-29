import torch
def scatter_pillars_to_bev(pillar_embeddings, pillar_coords, bev_shape):
    """
    Args:
        pillar_embeddings: (num_pillars, C) tensor from PillarEncoder
        pillar_coords: (num_pillars, 2) tensor (x, y)
        bev_shape: (H, W)
    Returns:
        bev: (C, H, W) tensor
    """
    C = pillar_embeddings.shape[1]
    H, W = bev_shape
    bev = torch.zeros((C, H, W), device=pillar_embeddings.device)

    x = pillar_coords[:, 0].clamp(0, W-1)
    y = pillar_coords[:, 1].clamp(0, H-1)

    bev[:, y, x] = pillar_embeddings.T
    return bev
