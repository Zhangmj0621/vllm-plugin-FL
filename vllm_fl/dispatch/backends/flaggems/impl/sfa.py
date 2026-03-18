from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import torch
import vllm.envs as envs_vllm
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MLAAttentionImpl  # type: ignore
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_fl import envs
from vllm_fl.vllmfl_config import get_vllm_fl_config
from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLState