# Copyright (c) 2025 BAAI. All rights reserved.

import os
import warnings
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.config import VllmConfig

class VllmFLConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
        
        self.layer_sharding = additional_config.get("layer_sharding", None)
        if self.layer_sharding is not None:
            logger.info_once(
                f"Linear layer sharding enabled with config: {self.layer_sharding}. "
                "Note: This feature works optimally with FLASHCOMM2 and DSA-CP enabled; "
                "using it without these features may result in significant performance degradation."
            )

_VLLM_FL_CONFIG: VllmFLConfig | None = None


def init_vllm_fl_config(vllm_config):
    additional_config = (
        vllm_config.additional_config
        if vllm_config.additional_config is not None
        else {}
    )
    refresh = additional_config.get("refresh", False) if additional_config else False
    global _VLLM_FL_CONFIG
    if _VLLM_FL_CONFIG is not None and not refresh:
        return _VLLM_FL_CONFIG
    _VLLM_FL_CONFIG = VllmFLConfig(vllm_config)
    return _VLLM_FL_CONFIG


def clear_vllm_fl_config():
    global _VLLM_FL_CONFIG
    _VLLM_FL_CONFIG = None


def get_vllm_fl_config():
    global _VLLM_FL_CONFIG
    if _VLLM_FL_CONFIG is None:
        raise RuntimeError(
            "VllmFL config is not initialized. Please call init_vllm_fl_config first."
        )
    return _VLLM_FL_CONFIG
