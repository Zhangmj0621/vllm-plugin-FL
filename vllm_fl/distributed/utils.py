import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group
from vllm.forward_context import get_forward_context

def all_gather_async(
    input: torch.Tensor, 
    group: GroupCoordinator, 
    output: torch.Tensor | None = None, 
    async_op: bool = True,
):
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)
