import torch
from type import Optional

from vllm.logger import logger
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group, get_world_group, init_model_parallel_group

from vllm_fl.vllmfl_config import get_vllm_fl_config
from vllm_fl.utils import enable_dsa_cp_with_layer_shard

_SHARD_WEIGHT: Optional[GroupCoordinator] = None

def init_fl_model_parallel(
    parallel_config: ParallelConfig,
):
    assert torch.distributed.is_initialized(), "torch.distributed must be initialized before initializing vllm-fl specific communicator groups."
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size

    vllm_all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        parallel_config.prefill_context_parallel_size,
        global_tp_size,
    )
    pcp_size = parallel_config.prefill_context_parallel_size

    def create_shard_weight_group(module_tp_group_ranks: None) -> GroupCoordinator:
        # Argument module_tp_group_ranks: The module specific tensor parallel group.
        # There are three situations.
        # 1. If it is None, then the TP_size of the specific module is 1 and is replicated linear layer.
        # 2. If it is not None, and the module tp_group is same as the global tp_group.
        # 3. If it is not None, and the module tp_group is different from the global tp_group.(eg. flashcomm2_otp)
        group_ranks = []
        pp_group_ranks = vllm_all_ranks.transpose(2, 4).reshape(-1, global_pp_size)
        if module_tp_group_ranks is None:
            # If it is None, then the TP_size of this shard weight is 1.
            # In fact, group_ranks means ranks with same pp stage
            shard_weight_group_ranks = pp_group_ranks.transpose(0, 1).unbind(0)
            group_ranks = [x.tolist() for x in shard_weight_group_ranks]
        else:
            # combine standard tp group and non-standard tp group to build  shard_weight comm_group
            # In fact, group_ranks means ranks with same pp stage and same tp stage
            module_tp_tanspose_ranks = module_tp_group_ranks.transpose(0, 1)
            G = world_size // (global_pp_size * module_tp_group_ranks.size(1))
            shard_weight_group_ranks = torch.stack(
                [t.view(global_pp_size, G) for t in module_tp_tanspose_ranks], dim=1
            )
            group_ranks = shard_weight_group_ranks.view(-1, G).tolist()
        return init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="shard_weight",
        )

    if get_vllm_fl_config().layer_sharding is not None:
        # TODO: if we support flashcomm2 like ascend, we can create shard weight group according to FC2
        global _SHARD_WEIGHT
        if enable_dsa_cp_with_layer_shard():
            # For dsa-cp, all shard layers are replicated.
            _SHARD_WEIGHT = create_shard_weight_group(None)
        else:
            # For standard tp, use global tp group_ranks
            tp_group_ranks = vllm_all_ranks.view(-1, global_tp_size)
            _SHARD_WEIGHT = create_shard_weight_group(tp_group_ranks)


def get_shard_weight_group() -> GroupCoordinator:
    assert (
        _SHARD_WEIGHT is not None
    ), "output shard weight parallel group is not initialized"
    return _SHARD_WEIGHT
