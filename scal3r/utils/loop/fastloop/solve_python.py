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

    rows, cols, data = [], [], []
    ii_np = ii.numpy()
    jj_np = jj.numpy()
    J_Ginv_i_np = J_Ginv_i.numpy()
    J_Ginv_j_np = J_Ginv_j.numpy()

    for edge in range(num_edges):
        i = ii_np[edge]
        j = jj_np[edge]
        if i == j:
            raise ValueError("Self-edges are not allowed")

        for row in range(7):
            for col in range(7):
                row_idx = edge * 7 + row
                rows.append(row_idx)
                cols.append(i * 7 + col)
                data.append(J_Ginv_i_np[edge, row, col])

                rows.append(row_idx)
                cols.append(j * 7 + col)
                data.append(J_Ginv_j_np[edge, row, col])

    J = coo_matrix((data, (rows, cols)), shape=(num_edges * 7, num_nodes * 7)).tocsc()
    b_vec = -J.T @ res_vec
    A_mat = J.T @ J
    diag = A_mat.diagonal()
    A_mat.setdiag(diag * (1.0 + lm) + ep)

    freen_total = freen * 7
    delta = solve_sparse(A_mat.tocsc(), b_vec, freen_total)
    return torch.from_numpy(delta.astype(np.float32)).view(num_nodes, 7).to(device)
