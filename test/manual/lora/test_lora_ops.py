import random
import unittest

import torch
from utils import reference_sgmv_expand, reference_sgmv_shrink

from sglang.srt.lora.torch_ops.lora_ops import sgemm_lora_a_fwd, sgemm_lora_b_fwd
from sglang.test.test_utils import CustomTestCase


class TestLoraOps(CustomTestCase):
    def test_sgemm_lora_a_fwd(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        inputs = torch.randn(batch_size, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(lora_scaling, dtype=torch.float16, device="cpu")

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        actual_output = sgemm_lora_a_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))


    def test_sgemm_lora_b_fwd(self):
        batch_size = 2
        output_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        inputs = torch.randn(batch_size, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        seq_len_tensor = torch.ones(batch_size, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(num_loras, (batch_size,), dtype=torch.int32,  device="cpu")
        slice_offsets = torch.tensor(
            [0, output_dim], dtype=torch.int32, device="cpu"
        )

        expect_output = reference_sgmv_expand(
            inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, lora_ranks_tensor, slice_offsets
        )

        actual_output = sgemm_lora_b_fwd(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))


    def test_sgemm_lora_a_fwd_expand(self):
        batch_size = 2
        input_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        possible_lora_scaling = [0.25, 0.5, 1.0, 2.0, 4.0]
        lora_scaling = random.sample(
            possible_lora_scaling,
            counts=[num_loras] * len(possible_lora_scaling),
            k=num_loras,
        )

        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )

        seq_len = sum(seq_len_tensor)

        inputs = torch.randn(seq_len, input_dim, dtype=dtype)
        lora_a_weights = torch.randn(num_loras, max_lora_rank, input_dim, dtype=dtype)
        lora_indices_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_scaling_tensor = torch.tensor(lora_scaling, dtype=torch.float16, device="cpu")

        expect_output = reference_sgmv_shrink(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        actual_output = sgemm_lora_a_fwd(
            inputs,
            lora_a_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            lora_scaling_tensor,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))


    def test_sgemm_lora_b_fwd_expand(self):
        batch_size = 2
        output_dim = 1024
        num_loras = 3
        dtype = torch.float32

        possible_lora_ranks = [8, 16, 32, 64, 128, 256]
        lora_ranks = random.sample(
            possible_lora_ranks,
            counts=[num_loras] * len(possible_lora_ranks),
            k=num_loras,
        )

        max_lora_rank = max(lora_ranks)

        seq_len_tensor = torch.randint(
            num_loras, (batch_size,), dtype=torch.int32, device="cpu"
        )

        seq_len = sum(seq_len_tensor)

        inputs = torch.randn(seq_len, max_lora_rank, dtype=dtype)
        lora_b_weights = torch.randn(num_loras, output_dim, max_lora_rank, dtype=dtype)
        lora_ranks_tensor = torch.tensor(lora_ranks, dtype=torch.int32, device="cpu")
        lora_indices_tensor = torch.randint(num_loras, (batch_size,), dtype=torch.int32,  device="cpu")
        slice_offsets = torch.tensor(
            [0, output_dim], dtype=torch.int32, device="cpu"
        )

        expect_output = reference_sgmv_expand(
            inputs, lora_b_weights, lora_indices_tensor, seq_len_tensor, lora_ranks_tensor, slice_offsets
        )

        actual_output = sgemm_lora_b_fwd(
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            seq_len_tensor,
            lora_ranks_tensor,
            slice_offsets,
        )

        self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_bgmv_expand(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))

    #     selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
    #     selected_loras = selected_loras.squeeze(dim=1)
    #     inputs = inputs.to(dtype=dtype)
    #     outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    #     limit = batch_size
    #     common_len = min(outputs.shape[1], output_dim)
    #     expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
    #     expect_output[:, :common_len] = outputs[:limit, :common_len]

    #     actual_output = sgemm_lora_b_fwd(
    #         inputs,
    #         lora_b_weights,
    #         lora_indices_tensor,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_bgmv_expand_add_residual(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))

    #     selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
    #     selected_loras = selected_loras.squeeze(dim=1)
    #     inputs = inputs.to(dtype=dtype)
    #     outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    #     limit = batch_size
    #     common_len = min(outputs.shape[1], output_dim)
    #     expect_output = torch.randn(batch_size, output_dim, dtype=dtype)
    #     actual_output = expect_output.clone()

    #     expect_output[:, :common_len] += outputs[:limit, :common_len]

    #     actual_output = sgemm_lora_b_fwd(
    #         inputs,
    #         lora_b_weights,
    #         actual_output,
    #         lora_indices_tensor,
    #         add_inputs=True,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_sgmv_shrink(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_a_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     seq_len_tensor = torch.ones(batch_size, dtype=torch.int32)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
    #     possible_scaling = [.25f, .5f, 1.f, 2.f, 4.f]
    #     scaling_tensor = random.sample(possible_scaling, counts=[num_loras] * len(possible_lora_ranks), k=num_loras)

    #     total_seq_len, _ = inputs.shape
    #     exploded_indices = torch.repeat_interleave(
    #         lora_indices_tensor, seq_len_tensor, output_size=total_seq_len
    #     )
    #     expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
    #     bgmv_shrink(inputs, lora_a_weights, expect_output, exploded_indices, scaling)

    #     actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
    #     sgmv_shrink(
    #         inputs,
    #         lora_a_weights,
    #         actual_output,
    #         seq_len_tensor,
    #         lora_indices_tensor,
    #         scaling,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_bgmv_shrink(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_a_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
    #     scaling = 0.9

    #     selected_loras = lora_a_weights[lora_indices_tensor].to(dtype=dtype)
    #     inputs = inputs.to(dtype=dtype)
    #     outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    #     expect_output = torch.zeros(batch_size, output_dim, dtype=dtype)
    #     expect_output[:, : outputs.shape[1]] = scaling * outputs[:]

    #     actual_output = torch.zeros(batch_size, output_dim, dtype=dtype)
    #     bgmv_shrink(
    #         inputs,
    #         lora_a_weights,
    #         actual_output,
    #         lora_indices_tensor,
    #         scaling=scaling,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_sgmv_expand_slice(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     output_dim_slice = 12
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     seq_len_tensor = torch.ones(batch_size, dtype=torch.int32)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
    #     slice_offset = 2
    #     slice_size = 6
    #     add_inputs = False

    #     total_seq_len, _ = inputs.shape
    #     exploded_indices = torch.repeat_interleave(
    #         lora_indices_tensor, seq_len_tensor, output_size=total_seq_len
    #     )
    #     expect_output = torch.randn(batch_size, output_dim_slice, dtype=dtype)
    #     actual_output = expect_output.clone()
    #     bgmv_expand_slice(
    #         inputs,
    #         lora_b_weights,
    #         expect_output,
    #         exploded_indices,
    #         slice_offset,
    #         slice_size,
    #         add_inputs,
    #     )

    #     sgmv_expand_slice(
    #         inputs,
    #         lora_b_weights,
    #         actual_output,
    #         seq_len_tensor,
    #         lora_indices_tensor,
    #         slice_offset,
    #         slice_size,
    #         add_inputs,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_bgmv_expand_slice(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     output_dim_slice = 12
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
    #     slice_offset = 2
    #     slice_size = 6

    #     selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
    #     inputs = inputs.to(dtype=dtype)
    #     outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    #     expect_output = torch.zeros(batch_size, output_dim_slice, dtype=dtype)
    #     expect_output[:, slice_offset : slice_offset + slice_size] = outputs[:]

    #     actual_output = torch.zeros(batch_size, output_dim_slice, dtype=dtype)
    #     bgmv_expand_slice(
    #         inputs,
    #         lora_b_weights,
    #         actual_output,
    #         lora_indices_tensor,
    #         slice_offset,
    #         slice_size,
    #         add_inputs=False,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))

    # def test_bgmv_expand_slice_add_residual(self):
    #     batch_size = 2
    #     input_dim = 4
    #     output_dim = 6
    #     output_dim_slice = 12
    #     num_loras = 3
    #     dtype = torch.float32

    #     inputs = torch.randn(batch_size, input_dim, dtype=dtype)
    #     lora_b_weights = torch.randn(num_loras, output_dim, input_dim, dtype=dtype)
    #     lora_indices_tensor = torch.randint(0, num_loras, (batch_size,))
    #     slice_offset = 2
    #     slice_size = 6

    #     selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=dtype)
    #     inputs = inputs.to(dtype=dtype)
    #     outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    #     expect_output = torch.randn(batch_size, output_dim_slice, dtype=dtype)
    #     actual_output = expect_output.clone()
    #     expect_output[:, slice_offset : slice_offset + slice_size] += outputs[:]

    #     bgmv_expand_slice(
    #         inputs,
    #         lora_b_weights,
    #         actual_output,
    #         lora_indices_tensor,
    #         slice_offset,
    #         slice_size,
    #         add_inputs=True,
    #     )

    #     self.assertTrue(torch.allclose(actual_output, expect_output))


if __name__ == "__main__":
    unittest.main()
