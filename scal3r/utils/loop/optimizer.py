import torch
import numpy as np
import pypose as pp

from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

from scal3r.utils.loop.fastloop.solve_python import solve_system_py

cpp_version = False
try:
    import sim3solve

    cpp_version = True
except Exception:
    sim3solve = None


class Sim3LoopOptimizer:
    def __init__(
        self,
        max_iterations: int = 30,
        lambda_init: float = 1e-6,
        device: str = "cpu",
    ):
        self.max_iterations = max_iterations
        self.lambda_init = lambda_init
        self.device = device
        self.solve_system_version = "cpp" if cpp_version else "python"

    def numpy_to_pypose_sim3(self, scale: float, rotation: np.ndarray, translation: np.ndarray):
        quat = R.from_matrix(rotation).as_quat()
        data = np.concatenate([translation, quat, np.array([scale])])
        return pp.Sim3(torch.from_numpy(data).float().to(self.device))

    def pypose_sim3_to_numpy(self, sim3) -> Tuple[float, np.ndarray, np.ndarray]:
        data = sim3.data.cpu().numpy()
        translation = data[:3]
        quat = data[3:7]
        scale = data[7]
        rotation = R.from_quat(quat).as_matrix()
        return scale, rotation, translation

    def sequential_to_absolute_poses(self, sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]]):
        identity = pp.Sim3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], device=self.device))
        poses = [identity]
        current_pose = identity
        for scale, rotation, translation in sequential_transforms:
            current_pose = current_pose @ self.numpy_to_pypose_sim3(scale, rotation, translation)
            poses.append(current_pose)
        return torch.stack(poses)

    def absolute_to_sequential_transforms(self, absolute_poses):
        sequential = []
        for idx in range(absolute_poses.shape[0] - 1):
            rel_transform = absolute_poses[idx].Inv() @ absolute_poses[idx + 1]
            sequential.append(self.pypose_sim3_to_numpy(rel_transform))
        return sequential

    def build_loop_constraints(
        self,
        loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
    ):
        if not loop_constraints:
            empty = torch.empty(0, device=self.device)
            return pp.Sim3(torch.empty(0, 8, device=self.device)), empty.long(), empty.long()

        loop_transforms = []
        ii_loop = []
        jj_loop = []
        for i, j, (scale, rotation, translation) in loop_constraints:
            loop_transforms.append(self.numpy_to_pypose_sim3(scale, rotation, translation).data)
            ii_loop.append(i)
            jj_loop.append(j)

        dSloop = pp.Sim3(torch.stack(loop_transforms))
        ii_loop = torch.tensor(ii_loop, dtype=torch.long, device=self.device)
        jj_loop = torch.tensor(jj_loop, dtype=torch.long, device=self.device)
        return dSloop, ii_loop, jj_loop

    def residual(self, Ginv, input_poses, dSloop, ii, jj, jacobian: bool = False):
        def _residual(C, Gi, Gj):
            out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
            return out.Log().tensor()

        pred_inv_poses = pp.Sim3(input_poses).Inv()
        num_poses, _ = pred_inv_poses.shape

        if num_poses > 1:
            kk = torch.arange(1, num_poses, device=self.device)
            ll = kk - 1
            Ti = pred_inv_poses[kk]
            Tj = pred_inv_poses[ll]
            dSij = Tj @ Ti.Inv()
        else:
            kk = torch.empty(0, dtype=torch.long, device=self.device)
            ll = torch.empty(0, dtype=torch.long, device=self.device)
            dSij = pp.Sim3(torch.empty(0, 8, device=self.device))

        constants = torch.cat((dSij.data, dSloop.data), dim=0) if dSloop.shape[0] > 0 else dSij.data
        if constants.shape[0] > 0:
            constants = pp.Sim3(constants)
            iii = torch.cat((kk, ii))
            jjj = torch.cat((ll, jj))
            resid = _residual(constants, Ginv[iii], Ginv[jjj])
        else:
            iii = torch.empty(0, dtype=torch.long, device=self.device)
            jjj = torch.empty(0, dtype=torch.long, device=self.device)
            resid = torch.empty(0, device=self.device)

        if not jacobian:
            return resid

        if constants.shape[0] > 0:
            def batch_jacobian(func, x):
                def _func_sum(*args):
                    return func(*args).sum(dim=0)

                _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
                from einops import rearrange

                return rearrange(torch.stack((b, c)), "N O B I -> N B O I", N=2)

            J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
        else:
            J_Ginv_i = torch.empty(0, device=self.device)
            J_Ginv_j = torch.empty(0, device=self.device)

        return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

    def optimize(
        self,
        sequential_transforms: List[Tuple[float, np.ndarray, np.ndarray]],
        loop_constraints: List[Tuple[int, int, Tuple[float, np.ndarray, np.ndarray]]],
        max_iterations: int | None = None,
        lambda_init: float | None = None,
    ):
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        lambda_init = self.lambda_init if lambda_init is None else lambda_init

        input_poses = self.sequential_to_absolute_poses(sequential_transforms)
        dSloop, ii_loop, jj_loop = self.build_loop_constraints(loop_constraints)
        if len(loop_constraints) == 0:
            return sequential_transforms

        Ginv = pp.Sim3(input_poses).Inv().Log()
        lmbda = lambda_init
        residual_history = []

        for itr in range(max_iterations):
            resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = self.residual(
                Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True
            )
            if resid.numel() == 0:
                break

            current_cost = resid.square().mean().item()
            residual_history.append(current_cost)

            try:
                if self.solve_system_version == "cpp":
                    delta_pose, = sim3solve.solve_system(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
                else:
                    delta_pose = solve_system_py(
                        J_Ginv_i, J_Ginv_j, iii, jjj, resid, 0.0, lmbda, -1
                    )
            except Exception:
                break

            Ginv_tmp = Ginv + delta_pose
            new_resid = self.residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
            new_cost = new_resid.square().mean().item() if new_resid.numel() > 0 else float("inf")

            if new_cost < current_cost:
                Ginv = Ginv_tmp
                lmbda /= 2
            else:
                lmbda *= 2

            if (current_cost < 1e-5) and (itr >= 4) and len(residual_history) >= 5:
                if residual_history[-5] / residual_history[-1] < 1.5:
                    break

        optimized_absolute_poses = pp.Exp(Ginv).Inv()
        return self.absolute_to_sequential_transforms(optimized_absolute_poses)
