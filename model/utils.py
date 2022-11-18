import math

import torch


class OffsetGenerator():
    @classmethod
    def initialize(cls, n_patch_side, pad_size):

        grid_1d = torch.linspace(-1, 1, n_patch_side).to('cuda')

        if pad_size > 0:
            pad_dist = torch.cumsum((grid_1d[-1] - grid_1d[-2]).repeat(pad_size), dim=0)
            grid_1d = torch.cat([(-1 - pad_dist).flip(dims=[0]), grid_1d, 1 + pad_dist])
            n_patch_side += (pad_size * 2)
        n_tokens = n_patch_side ** 2

        grid_y = grid_1d.view(-1, 1).repeat(1, n_patch_side)
        grid_x = grid_1d.view(1, -1).repeat(n_patch_side, 1)
        grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)

        grid_q = grid.view(-1, 1, 2).repeat(1, n_tokens, 1)
        grid_k = grid.view(1, -1, 2).repeat(n_tokens, 1, 1)

        cls.qk_vec = grid_k - grid_q

    @classmethod
    def get_qk_vec(cls):
        return cls.qk_vec.clone()

