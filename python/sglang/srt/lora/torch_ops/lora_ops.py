from typing import Optional

import torch

from sglang.srt.lora.utils import LoRABatchInfo


def sgemm_lora_a_graph_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
    batch_info: Optional[LoRABatchInfo] = None,
):
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices
    bs = batch_info.bs

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )

    for lora_idx in range(num_loras):

        indices = torch.repeat_interleave(
            batch_info.weight_indices[:bs],
            batch_info.seg_lens[:bs],
            output_size=total_seq_len,
        )
        batch_token_mask = torch.where(indices == lora_idx, True, False).unsqueeze(1)

        x_seq = torch.where(batch_token_mask, inputs, 0)
        w_seq = weights[lora_idx]

        result = torch.mm(x_seq, w_seq.T)
        output.add_(scaling_tensor[lora_idx] * result)

    return output

def sgemm_lora_a_embedding_control_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
    batch_info: Optional[LoRABatchInfo] = None,
):
    total_seq_len = inputs.shape[0]
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, max_rank, _ = weights.shape

    output = torch.zeros(
        total_seq_len, max_rank, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:

            x_seq = inputs[token_offset : token_offset + seq_len]
            w_seq = weights[lora_idx, : rank, :]

            output[token_offset : token_offset + seq_len, : rank] = (
                torch.nn.functional.embedding(x_seq, w_seq.t())
            )

        token_offset += seq_len

    return output


def sgemm_lora_a_control_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
    batch_info: Optional[LoRABatchInfo] = None,
):
    total_seq_len, input_dim = inputs.shape
    if weights.numel() == 0:
        return torch.zeros(total_seq_len, 0, dtype=inputs.dtype, device=inputs.device)

    num_loras, weight_out_dim, _ = weights.shape
    max_rank = weight_out_dim // num_slices

    output = torch.zeros(
        total_seq_len, num_slices * max_rank, dtype=inputs.dtype, device=inputs.device
    )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank > 0:

            x_seq = inputs[token_offset : token_offset + seq_len, :]
            w_seq = weights[lora_idx, : num_slices * rank, :]

            result = torch.mm(x_seq, w_seq.T)
            output[token_offset : token_offset + seq_len, : num_slices * rank] = (
                scaling_tensor[lora_idx] * result
            )

        token_offset += seq_len

    return output


def sgemm_lora_b_graph_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
    batch_info: Optional[LoRABatchInfo] = None,
):
    total_seq_len, input_dim = inputs.shape
    num_loras, weight_out_dim, _ = weights.shape
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0

    if weights.numel() == 0:
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    num_slices = len(slice_offsets) - 1
    max_rank = input_dim // num_slices
    bs = batch_info.bs

    if base_output is not None:
        output = base_output
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    for lora_idx in range(num_loras):

        indices = torch.repeat_interleave(
            batch_info.weight_indices[:bs],
            batch_info.seg_lens[:bs],
            output_size=total_seq_len,
        )
        batch_token_mask = torch.where(indices == lora_idx, True, False).unsqueeze(1)

        for slice_idx in range(num_slices):
            slice_start_input = slice_idx * max_rank
            slice_end_input = (slice_idx + 1) * max_rank

            slice_start_output = slice_offsets[slice_idx]
            slice_end_output = slice_offsets[slice_idx + 1]

            x_slice = torch.where(
                batch_token_mask, inputs[:, slice_start_input:slice_end_input], 0
            )  # (total_seq_len, max_rank)
            w_slice = weights[
                lora_idx, slice_start_output:slice_end_output, :
            ]  # (slice_dim, max_rank)
            output[:, slice_start_output:slice_end_output].add_(
                torch.mm(x_slice, w_slice.T)
            )

    return output


def sgemm_lora_b_control_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
    batch_info: Optional[LoRABatchInfo] = None,
):
    total_seq_len, _ = inputs.shape
    num_loras, weight_out_dim, _ = weights.shape
    total_output_dim = slice_offsets[-1].item() if len(slice_offsets) > 0 else 0

    if weights.numel() == 0:
        return torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    num_slices = len(slice_offsets) - 1

    if base_output is not None:
        output = base_output
    else:
        output = torch.zeros(
            total_seq_len, total_output_dim, dtype=inputs.dtype, device=inputs.device
        )

    token_offset = 0
    for lora_idx, seq_len, rank in zip(
        weight_indices, seg_len_tensor, lora_ranks[weight_indices]
    ):
        if seq_len == 0:
            continue

        if rank == 0:
            token_offset += seq_len
            continue

        for slice_idx in range(num_slices):
            slice_start_input = slice_idx * rank
            slice_end_input = (slice_idx + 1) * rank

            slice_start_output = slice_offsets[slice_idx]
            slice_end_output = slice_offsets[slice_idx + 1]

            x_slice = inputs[
                token_offset : token_offset + seq_len :,
                slice_start_input:slice_end_input,
            ]  # (seq_len, rank)
            w_slice = weights[
                lora_idx, slice_start_output:slice_end_output, :rank
            ]  # (slice_dim, rank)

            output[
                token_offset : token_offset + seq_len,
                slice_start_output:slice_end_output,
            ].add_(torch.mm(x_slice, w_slice.T))

        token_offset += seq_len

    return output


def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    scaling_tensor: torch.Tensor,
    num_slices: int = 1,
    batch_info: Optional[LoRABatchInfo] = None,
):
    if batch_info.use_cuda_graph:
        return sgemm_lora_a_graph_fwd(
            inputs=inputs,
            weights=weights,
            weight_indices=weight_indices,
            seg_len_tensor=seg_len_tensor,
            lora_ranks=lora_ranks,
            scaling_tensor=scaling_tensor,
            num_slices=num_slices,
            batch_info=batch_info,
        )
    else:
        return sgemm_lora_a_control_fwd(
            inputs=inputs,
            weights=weights,
            weight_indices=weight_indices,
            seg_len_tensor=seg_len_tensor,
            lora_ranks=lora_ranks,
            scaling_tensor=scaling_tensor,
            num_slices=num_slices,
            batch_info=batch_info,
        )


def sgemm_lora_b_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    weight_indices: torch.Tensor,
    seg_len_tensor: torch.Tensor,
    lora_ranks: torch.Tensor,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
    batch_info: Optional[LoRABatchInfo] = None,
):
    if batch_info.use_cuda_graph:
        return sgemm_lora_b_graph_fwd(
            inputs=inputs,
            weights=weights,
            weight_indices=weight_indices,
            seg_len_tensor=seg_len_tensor,
            lora_ranks=lora_ranks,
            slice_offsets=slice_offsets,
            base_output=base_output,
            batch_info=batch_info,
        )
    else:
        return sgemm_lora_b_control_fwd(
            inputs=inputs,
            weights=weights,
            weight_indices=weight_indices,
            seg_len_tensor=seg_len_tensor,
            lora_ranks=lora_ranks,
            slice_offsets=slice_offsets,
            base_output=base_output,
            batch_info=batch_info,
        )

