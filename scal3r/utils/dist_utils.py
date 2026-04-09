import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ReduceOp, group


class _SingleProcessGroup:
    def num_groups_active(self):
        return 1

    def get_group_active(self, rank):
        return None

    def get_group_active_size(self, rank):
        return 1

    def get_group_active_local_rank(self, rank):
        return 0


def get_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if get_distributed() else 1


def get_local_size() -> int:
    return get_world_size()


def get_rank() -> int:
    return dist.get_rank() if get_distributed() else 0


def get_real_pgc():
    return _SingleProcessGroup()


@torch._dynamo.allow_in_graph
def ddp_allreduce(
    x: torch.Tensor,
    group=None,
):
    if not get_distributed():
        return x
    return all_reduce(x, op=dist.ReduceOp.SUM, group=group)


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    return _AllReduce.apply(op, group, tensor)


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone().contiguous()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)


def context_parallelism(
    w0_grad: torch.Tensor,
    w1_grad: torch.Tensor,
    w2_grad: torch.Tensor,
):
    if not get_distributed():
        return w0_grad, w1_grad, w2_grad

    pgc = get_real_pgc()
    if pgc.num_groups_active() == get_world_size():
        return w0_grad, w1_grad, w2_grad

    active_group = pgc.get_group_active(get_rank())
    w0_grad = ddp_allreduce(
        w0_grad,
        group=active_group,
    )
    w1_grad = ddp_allreduce(
        w1_grad,
        group=active_group,
    )
    w2_grad = ddp_allreduce(
        w2_grad,
        group=active_group,
    )
    return w0_grad, w1_grad, w2_grad


def get_group_info():
    pgc = get_real_pgc()
    return (
        pgc.num_groups_active(),
        pgc.get_group_active_size(get_rank()),
        pgc.get_group_active_local_rank(get_rank()),
    )
