import torch
import numpy as np

from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix, csc_matrix

def solve_sparse(A: csc_matrix, b: np.ndarray, freen: int) -> np.ndarray:
    if freen < 0:
        return spsolve(A, b)

    A_sub = A[:freen, :freen].tocsc()
    b_sub = b[:freen]
    delta_sub = spsolve(A_sub, b_sub)
    delta = np.zeros_like(b)
    delta[:freen] = delta_sub
    return delta


def solve_system_py(
    J_Ginv_i: torch.Tensor,
    J_Ginv_j: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    res: torch.Tensor,
    ep: float,
    lm: float,
    freen: int,
) -> torch.Tensor:
    device = res.device
    J_Ginv_i = J_Ginv_i.cpu()
    J_Ginv_j = J_Ginv_j.cpu()
    ii = ii.cpu()
    jj = jj.cpu()
    res = res.clone().cpu()

    num_edges = res.size(0)
    num_nodes = max(ii.max().item(), jj.max().item()) + 1
    res_vec = res.view(-1).numpy().astype(np.float64)

    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    # Vectorized sparse Jacobian construction
    edge_idx = np.arange(num_edges)
    # Row indices: each edge has a 7x7 block for i and a 7x7 block for j (49 entries each)
    row_offsets = np.repeat(np.arange(7), 7)  # [0,0,...,0,1,1,...,1,...,6,6,...,6]
    i_rows = np.repeat(edge_idx * 7, 49) + np.tile(row_offsets, num_edges)

    # Column indices
    col_offsets = np.tile(np.arange(7), 7)  # [0,1,...,6,0,1,...,6,...,0,1,...,6]
    i_cols = np.repeat(ii_np * 7, 49) + np.tile(col_offsets, num_edges)
    j_cols = np.repeat(jj_np * 7, 49) + np.tile(col_offsets, num_edges)

    # Stack i-block and j-block
    all_rows = np.concatenate([i_rows, i_rows])
    all_cols = np.concatenate([i_cols, j_cols])
    all_data = np.concatenate([J_Ginv_i_np.reshape(-1), J_Ginv_j_np.reshape(-1)])

    J = coo_matrix((all_data, (all_rows, all_cols)), shape=(num_edges * 7, num_nodes * 7)).tocsc()
    b_vec = -J.T @ res_vec
    A_mat = J.T @ J
    diag = A_mat.diagonal()
    A_mat.setdiag(diag * (1.0 + lm) + ep)

    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)
    return torch.from_numpy(delta.astype(np.float32)).view(num_nodes, 7).to(device)
