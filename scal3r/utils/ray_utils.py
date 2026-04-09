import torch

from scal3r.utils.math_utils import normalize, torch_inverse_3x3


def get_rays_from_ij(
    i: torch.Tensor,
    j: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    is_inv_K: bool = False,
    use_z_depth: bool = False,
    correct_pix: bool = True,
    ret_coord: bool = False,
):
    # i: B, P or B, H, W or P or H, W
    # j: B, P or B, H, W or P or H, W
    # K: B, 3, 3
    # R: B, 3, 3
    # T: B, 3, 1
    nb_dim = len(K.shape[:-2])
    np_dim = len(i.shape[nb_dim:])
    if not is_inv_K:
        invK = torch_inverse_3x3(K.float()).type(K.dtype)
    else:
        invK = K
    ray_o = -R.mT @ T

    for _ in range(np_dim):
        invK = invK.unsqueeze(-3)
    invK = invK.expand(i.shape + (3, 3))
    for _ in range(np_dim):
        R = R.unsqueeze(-3)
    R = R.expand(i.shape + (3, 3))
    for _ in range(np_dim):
        T = T.unsqueeze(-3)
    T = T.expand(i.shape + (3, 1))
    for _ in range(np_dim):
        ray_o = ray_o.unsqueeze(-3)
    ray_o = ray_o.expand(i.shape + (3, 1))

    if correct_pix:
        i, j = i + 0.5, j + 0.5
    else:
        i, j = i.float(), j.float()

    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)[..., None]
    pixel_camera = invK @ xy1
    pixel_world = R.mT @ (pixel_camera - T)

    pixel_world = pixel_world[..., 0]
    ray_o = ray_o[..., 0]
    ray_d = pixel_world - ray_o
    if not use_z_depth:
        ray_d = normalize(ray_d)

    if not ret_coord:
        return ray_o, ray_d
    elif correct_pix:
        return ray_o, ray_d, (torch.stack([i, j], dim=-1) - 0.5).long()
    else:
        return ray_o, ray_d, torch.stack([i, j], dim=-1).long()


def get_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    is_inv_K: bool = False,
    z_depth: bool = False,
    correct_pix: bool = True,
    ret_coord: bool = False,
):
    # calculate the world coordinates of pixels
    i, j = torch.meshgrid(
        torch.arange(H, dtype=R.dtype, device=R.device),
        torch.arange(W, dtype=R.dtype, device=R.device),
        indexing="ij",
    )
    bss = K.shape[:-2]
    for _ in range(len(bss)):
        i, j = i[None], j[None]
    i, j = i.expand(bss + i.shape[len(bss):]), j.expand(bss + j.shape[len(bss):])
    return get_rays_from_ij(i, j, K, R, T, is_inv_K, z_depth, correct_pix, ret_coord)
