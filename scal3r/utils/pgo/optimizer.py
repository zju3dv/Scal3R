import os
import numpy as np

from scal3r.utils.data_utils import export_pts
from scal3r.utils.pgo.utils import SerializableMixin


class PoseGraphOptimizer(SerializableMixin):
    def __init__(self):
        self.submaps = {}
        self.sim3_constraints = []

    def add_submap(self, submap):
        self.submaps[submap.submap_id] = submap

    def get_submap(self, submap_id):
        return self.submaps[submap_id]

    def add_sim3_constraint(self, constraint):
        self.sim3_constraints.append(constraint)

    def save_ply(self, points, output_path):
        export_pts(points[:, :3], filename=output_path)

    def save_pointclouds(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        global_cloudpoints_list = []
        for key, submap in self.submaps.items():
            global_cloudpoints = submap.get_global_xyz()
            self.save_ply(global_cloudpoints, f"{output_path}/{key}.ply")
            global_cloudpoints_list.append(global_cloudpoints)

        # global_cloudpoints_np = np.concatenate(global_cloudpoints_list, axis=0)
        # self.save_ply(global_cloudpoints_np, f"{output_path}/global_cloudpoints.ply")

    def optimize_pypose(self, iters=100, lr=1e-2, anchor_weight=100.0):
        """ Optimize the pose graph using pypose library """
        import torch
        import pypose as pp
        from torch import nn
        import pypose.optim.solver as ppos
        import pypose.optim.strategy as ppost
        from pypose.optim.scheduler import StopOnPlateau

        class PoseGraph(nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.nodes = pp.Parameter(nodes)

            def forward(self, edges, poses):
                node1 = self.nodes[edges[..., 0]]
                node2 = self.nodes[edges[..., 1]]
                error = poses.Inv() @ node1.Inv() @ node2
                return error.Log().tensor()
        
        poses = torch.stack([
            pp.mat2Sim3(submap.global_pose) for submap in self.submaps.values()
        ])
        graph = PoseGraph(poses)

        solver = ppos.Cholesky()
        strategy = ppost.TrustRegion(radius=10000)
        optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-6, vectorize=False)
        scheduler = StopOnPlateau(optimizer, steps=iters, patience=3, decreasing=1e-3, verbose=True)

        edges = []
        meas = []
        for c in self.sim3_constraints:
            edges.append([c.submap_id1, c.submap_id2])
            meas.append(pp.mat2Sim3(c.sim3))
        edges = np.array(edges)
        meas = torch.stack(meas)

        scheduler.optimize(input=(edges, meas), weight=None)

        for key, submap in self.submaps.items():
            submap.global_pose = poses[key].matrix().cpu().numpy()
