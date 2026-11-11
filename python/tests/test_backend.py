# mpirun -np 2 ~/venvs/hxgemm/bin/python -m pytest -q python/tests
import os

import pytest
import torch
import torch.distributed as dist

import hxccl_torch  # noqa: F401  registers "hxccl" and fills RANK/WORLD_SIZE from OpenMPI


@pytest.fixture(scope="module", autouse=True)
def groups():
    dist.init_process_group("hxccl")
    gloo = dist.new_group(backend="gloo")
    yield gloo
    dist.destroy_process_group()


def rank_tensor(n, dtype=torch.float32):
    r = dist.get_rank()
    return (torch.arange(n, dtype=torch.float64) % 97 + r).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.AVG])
@pytest.mark.parametrize("n", [1, 1000, 1 << 20])
def test_all_reduce_matches_gloo(groups, dtype, op, n):
    if op == dist.ReduceOp.AVG and not dtype.is_floating_point:
        pytest.skip("gloo has no integer AVG")
    t = rank_tensor(n, dtype).cuda()
    dist.all_reduce(t, op=op)
    ref = rank_tensor(n, dtype)
    dist.all_reduce(ref, op=op, group=groups)
    if dtype.is_floating_point:
        torch.testing.assert_close(t.cpu(), ref)
    else:
        assert torch.equal(t.cpu(), ref)


def test_all_gather_into_tensor(groups):
    t = rank_tensor(4097).cuda()
    out = torch.empty(4097 * dist.get_world_size(), device="cuda")
    dist.all_gather_into_tensor(out, t)
    ref = torch.empty(4097 * dist.get_world_size())
    dist.all_gather_into_tensor(ref, rank_tensor(4097), group=groups)
    assert torch.equal(out.cpu(), ref)


def test_reduce_scatter_tensor():
    w = dist.get_world_size()
    t = rank_tensor(1000 * w).cuda()
    out = torch.empty(1000, device="cuda")
    dist.reduce_scatter_tensor(out, t)
    full = sum((torch.arange(1000 * w, dtype=torch.float64) % 97 + r) for r in range(w))
    r = dist.get_rank()
    torch.testing.assert_close(out.cpu(), full[r * 1000:(r + 1) * 1000].float())


def test_broadcast_send_recv_barrier():
    t = torch.full((10,), float(dist.get_rank()), device="cuda")
    dist.broadcast(t, src=0)
    assert torch.equal(t.cpu(), torch.zeros(10))
    w, r = dist.get_world_size(), dist.get_rank()
    if w >= 2 and r < 2:
        buf = torch.full((5,), float(r + 1), device="cuda")
        if r == 0:
            dist.send(buf, dst=1)
        else:
            dist.recv(buf, src=0)
            assert torch.equal(buf.cpu(), torch.ones(5))
    dist.barrier()


def test_rejects_non_contiguous():
    t = torch.ones(4, 4, device="cuda").t()
    with pytest.raises((ValueError, RuntimeError)):
        dist.all_reduce(t)
