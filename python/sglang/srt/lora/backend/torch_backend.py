from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.torch_ops import sgemm_lora_a_fwd, sgemm_lora_b_fwd
from sglang.srt.lora.utils import LoRABatchInfo, generate_sequence_lengths
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class TorchNativeLoRABatchInfo(LoRABatchInfo):
    lora_ranks_cpu: Optional[torch.Tensor] = None
    seg_indptr_cpu: Optional[torch.Tensor] = None
    seg_lens_cpu: Optional[torch.Tensor] = None
    weight_indices_cpu: Optional[torch.Tensor] = None
    permutation_reordered: Optional[torch.Tensor] = None
    hash_result: Optional[int] = None


class TorchNativeLoRABackend(BaseLoRABackend):
    name = "torch_native"

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_loras_per_batch, device)

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        total_seq_len, _ = x.shape
        if sum(self.batch_info.lora_ranks_cpu[self.batch_info.weight_indices_cpu]):
            return torch.zeros(total_seq_len, 0, dtype=x.dtype, device=x.device)
        permutation = self.cuda_graph_batch_info.permutation
        permutation_reordered = self.cuda_graph_batch_info.permutation_reordered
        reordered_x = x[permutation]

        output_tensor = sgemm_lora_a_fwd(
            inputs=reordered_x,
            weights=weights,
            seq_len_tensor=self.batch_info.seq_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling=self.batch_info.scalings,
            num_slices=1,
        )

        return output_tensor[permutation_reordered]

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        total_seq_len, _ = x.shape
        _, weight_out_dim, _ = weights.shape

        if sum(self.batch_info.lora_ranks_cpu[self.batch_info.weight_indices_cpu]):
            return (
                torch.zeros(
                    total_seq_len, weight_out_dim, dtype=x.dtype, device=x.device
                )
                if base_output is None
                else base_output
            )

        permutation = self.cuda_graph_batch_info.permutation
        permutation_reordered = self.cuda_graph_batch_info.permutation_reordered
        reordered_x = x[permutation]

        output_tensor = sgemm_lora_b_fwd(
            inputs=reordered_x,
            weights=weights,
            seg_lens_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=torch.tensor(
                [0, weight_out_dim], dtype=torch.int32, device="cpu"
            ),
            base_output=base_output,
        )

        return output_tensor[permutation_reordered]

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset: torch.Tensor,
        output_offset_cpu: torch.Tensor,
        max_qkv_out_dim: int,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        num_slices = 3
        assert isinstance(qkv_lora_b, torch.Tensor)

        total_seq_len, _ = x.shape
        _, weight_intermediate_dim, _ = qkv_lora_a.shape
        _, weight_out_dim, _ = qkv_lora_b.shape
        max_rank = weight_intermediate_dim // num_slices

        if sum(self.batch_info.lora_ranks_cpu[self.batch_info.weight_indices_cpu]):
            return (
                torch.zeros(
                    total_seq_len, weight_out_dim, dtype=x.dtype, device=x.device
                )
                if base_output is None
                else base_output
            )

        permutation = self.cuda_graph_batch_info.permutation
        permutation_reordered = self.cuda_graph_batch_info.permutation_reordered
        reordered_x = x[permutation]

        lora_a_output = sgemm_lora_a_fwd(
            inputs=reordered_x,
            weights=qkv_lora_a,
            seq_len_tensor=self.batch_info.seq_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling=self.batch_info.scalings,
            num_slices=3,
        )

        output_tensor = sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=qkv_lora_b,
            seg_lens_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=output_offset_cpu,
            base_output=base_output,
        )

        return output_tensor[permutation_reordered]

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        num_slices = 2
        assert isinstance(gate_up_lora_b, torch.Tensor)

        total_seq_len, _ = x.shape
        _, weight_intermediate_dim, _ = gate_up_lora_a.shape
        _, weight_out_dim, _ = gate_up_lora_b.shape
        slice_size = weight_out_dim // num_slices
        max_rank = weight_intermediate_dim // num_slices

        if sum(self.batch_info.lora_ranks_cpu[self.batch_info.weight_indices_cpu]):
            return (
                torch.zeros(
                    total_seq_len, weight_out_dim, dtype=x.dtype, device=x.device
                )
                if base_output is None
                else base_output
            )

        permutation = self.cuda_graph_batch_info.permutation
        permutation_reordered = self.cuda_graph_batch_info.permutation_reordered
        reordered_x = x[permutation]

        lora_a_output = sgemm_lora_a_fwd(
            inputs=reordered_x,
            weights=gate_up_lora_a,
            seq_len_tensor=self.batch_info.seq_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            scaling=self.batch_info.scalings,
            num_slices=3,
        )

        output_tensor = sgemm_lora_b_fwd(
            inputs=lora_a_output,
            weights=gate_up_lora_b,
            seg_lens_tensor=self.batch_info.seg_lens_cpu,
            lora_ranks=self.batch_info.lora_ranks_cpu,
            slice_offsets=torch.tensor(
                [0, slice_size, weight_out_dim], dtype=torch.int32, device="cpu"
            ),
            base_output=base_output,
        )

        return output_tensor[permutation_reordered]

    def init_cuda_graph_batch_info(
        self,
        max_bs_in_cuda_graph: int,
        num_tokens_per_bs: int,
    ):
        max_num_tokens = max_bs_in_cuda_graph * num_tokens_per_bs
        with torch.device("cuda"):
            self.cuda_graph_batch_info = TorchNativeLoRABatchInfo(
                use_cuda_graph=True,
                bs=max_bs_in_cuda_graph,
                num_segments=self.max_loras_per_batch,
                seg_lens=torch.full(
                    (max_bs_in_cuda_graph,), num_tokens_per_bs, dtype=torch.int32
                ),
                seg_indptr=torch.zeros(max_bs_in_cuda_graph + 1, dtype=torch.int32),
                weight_indices=torch.zeros(max_bs_in_cuda_graph, dtype=torch.int32),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
                permutation=torch.zeros(max_num_tokens, dtype=torch.int32),
                permutation_reordered=torch.zeros(max_num_tokens, dtype=torch.int32),
                max_len=num_tokens_per_bs,
            )

            # Initialize seg_indptr for CUDA graph as they remain constant
            # across batches.
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:max_bs_in_cuda_graph],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : max_bs_in_cuda_graph + 1],
            )

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        use_cuda_graph: bool,
    ):
        hash_result = hash(
            (
                forward_batch.forward_mode,
                (
                    (forward_batch.batch_size, forward_batch.spec_info.draft_token_num)
                    if forward_batch.forward_mode.is_target_verify()
                    else (
                        tuple(forward_batch.extend_seq_lens_cpu)
                        if forward_batch.forward_mode.is_extend()
                        else (forward_batch.batch_size)
                    )
                ),
                tuple(weight_indices),
                tuple(lora_ranks),
                tuple(scalings),
                use_cuda_graph,
            )
        )

        if (
            hasattr(self, "batch_info")
            and self.batch_info is not None
            and self.batch_info.hash_result is not None
        ):
            if self.batch_info.hash_result == hash_result:
                return

        original_seq_lens_cpu = generate_sequence_lengths(forward_batch)
        original_weight_indices_tensor = torch.tensor(
            weight_indices, dtype=torch.int32, device="cpu"
        )

        permutation = (
            torch.argsort(
                torch.repeat_interleave(
                    original_weight_indices_tensor, original_seq_lens_cpu
                ),
                stable=True,
            )
            .to(dtype=torch.int32)
            .pin_memory()
        )
        permutation_reordered = (
            torch.argsort(permutation, stable=True).to(dtype=torch.int32).pin_memory()
        )

        unique_weight_indices_tensor, inverse_ = torch.unique(
            original_weight_indices_tensor, sorted=True, return_inverse=True
        )

        seg_lens_cpu = (
            torch.zeros(self.max_loras_per_batch, dtype=torch.int32, device="cpu")
            .scatter_add_(
                0,
                original_weight_indices_tensor.to(dtype=torch.int64),
                original_seq_lens_cpu,
            )
            .pin_memory()
        )

        seg_indptr_cpu = torch.zeros(
            (len(seg_lens_cpu) + 1,), dtype=torch.int32, pin_memory=True
        )
        seg_indptr_cpu[1:] = torch.cumsum(seg_lens_cpu, dim=0)

        # Use pinned memory to avoid synchronizations during host-to-device transfer
        weight_indices_tensor = unique_weight_indices_tensor.pin_memory()
        lora_ranks_tensor = torch.tensor(
            lora_ranks, dtype=torch.int32, pin_memory=True, device="cpu"
        )
        scalings_tensor = torch.tensor(
            scalings, dtype=torch.float, pin_memory=True, device="cpu"
        )

        bs = forward_batch.batch_size

        if use_cuda_graph:
            assert (
                self.cuda_graph_batch_info is not None
            ), "CUDA Graph batch info is not initialized."
            batch_info = self.cuda_graph_batch_info
            batch_info.bs = forward_batch.batch_size
            batch_info.num_segments = forward_batch.batch_size
        else:
            max_len = seg_lens_cpu
            max_seq_len = sum(seg_lens_cpu)

            batch_info = TorchNativeLoRABatchInfo(
                bs=forward_batch.batch_size,
                num_segments=forward_batch.batch_size,
                max_len=max_len,
                use_cuda_graph=False,
                seg_lens=torch.empty((bs,), dtype=torch.int32, device=self.device),
                seg_indptr=torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                ),
                weight_indices=torch.empty(
                    (bs,), dtype=torch.int32, device=self.device
                ),
                lora_ranks=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.int32, device=self.device
                ),
                scalings=torch.empty(
                    (self.max_loras_per_batch,), dtype=torch.float, device=self.device
                ),
                permutation=torch.zeros(
                    max_seq_len, dtype=torch.int32, device=self.device
                ),
                permutation_reordered=torch.zeros(
                    max_seq_len, dtype=torch.int32, device=self.device
                ),
            )

        # Copy to device asynchronously
        batch_info.lora_ranks[: self.max_loras_per_batch].copy_(
            lora_ranks_tensor, non_blocking=True
        )
        batch_info.scalings[: self.max_loras_per_batch].copy_(
            scalings_tensor, non_blocking=True
        )
        batch_info.weight_indices[:bs].copy_(weight_indices_tensor, non_blocking=True)
        batch_info.seg_indptr[: (bs + 1)].copy_(seg_indptr_cpu, non_blocking=True)
        batch_info.seg_lens[:(bs)].copy_(seg_lens_cpu, non_blocking=True)  # spelling
        batch_info.permutation[: len(permutation)].copy_(permutation, non_blocking=True)
        batch_info.permutation_reordered[: len(permutation_reordered)].copy_(
            permutation_reordered, non_blocking=True
        )

        batch_info.lora_ranks_cpu = lora_ranks_tensor
        batch_info.seg_indptr_cpu = seg_indptr_cpu
        batch_info.seg_lens_cpu = seg_lens_cpu
        batch_info.weight_indices_cpu = weight_indices_tensor
        batch_info.hash_result = hash_result

        self.batch_info = batch_info
