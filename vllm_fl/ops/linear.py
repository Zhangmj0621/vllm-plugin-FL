"""Custom Linear classes for DSA-CP and Sequence Parallelism.

These replace vLLM's native ColumnParallelLinear, RowParallelLinear, and
MergedColumnParallelLinear via CustomOp.register_oot(). The only difference
from the vLLM originals is the custom_op dispatch:

1. get_parallel_op() is called at __init__ time to select a custom op
   based on the layer's prefix and enabled features.
2. If custom_op has tp_size=1 (ShardedCP), disable_tp is forced True
   so vLLM creates full (unsharded) weights.
3. In forward(), if custom_op is present, it delegates to custom_op.apply()
   which handles communication (all_gather, reduce_scatter, or none).
"""

from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_fl.ops.linear_op import get_parallel_op


class FLColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: QuantizationConfig | None = None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, _, _ = get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        if self.custom_op is not None and self.custom_op.tp_size == 1:
            disable_tp = True

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            output_sizes=output_sizes,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


class FLMergedColumnParallelLinear(MergedColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, _, _ = get_parallel_op(
            disable_tp, prefix, self, "column"
        )
        if self.custom_op is not None and self.custom_op.tp_size == 1:
            disable_tp = True

        super().__init__(
            input_size=input_size,
            output_sizes=output_sizes,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)


class FLRowParallelLinear(RowParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype=None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.custom_op, _, _ = get_parallel_op(
            disable_tp, prefix, self, "row"
        )
        if self.custom_op is not None and self.custom_op.tp_size == 1:
            disable_tp = True

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        if self.custom_op is not None:
            self.custom_op.update_attrs()

    def forward(self, input_):
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        return super().forward(input_)
