# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from https://github.com/AntixK/PyTorch-Model-Compare
# --------------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np
from .misc import AXES_FONT_SIZE, DPI, BASE_FONT_SIZE, FONT_NAME, AXES_FONT_SIZE, add_colorbar


class RepCmpAnalyzer(object):
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.hsic_matrix = torch.zeros(N, M, 3)
        self.num_batches = 0

    def append(self, features1, features2):
        # features1: num_layers * b * num_tokens * c
        # features2: num_layers * b * num_tokens * c
        assert len(features1) == self.N
        assert len(features2) == self.M
        for i, feat1 in enumerate(features1):
            X = feat1.flatten(1)
            K = X @ X.t()
            K.fill_diagonal_(0.0)
            self.hsic_matrix[i, :, 0] += self._HSIC(K, K)
            for j, feat2 in enumerate(features2):
                Y = feat2.flatten(1)
                L = Y @ Y.t()
                L.fill_diagonal_(0)
                assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                self.hsic_matrix[i, j, 1] += self._HSIC(K, L)
                self.hsic_matrix[i, j, 2] += self._HSIC(L, L)
        assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
        self.num_batches += 1
        
    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(K.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    def get(self):
        hsic_matrix = self.hsic_matrix / self.num_batches
        hsic_matrix = hsic_matrix[:, :, 1] / (hsic_matrix[:, :, 0].sqrt() * hsic_matrix[:, :, 2].sqrt())
        return hsic_matrix

    def plot_rep_cmp(self, save_path, name1, name2):
        hsic_matrix = self.get()
        plt.rcParams['font.size'] = BASE_FONT_SIZE
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = [FONT_NAME]
        fig, ax = plt.subplots(dpi=DPI)
        im = ax.imshow(hsic_matrix, origin='lower', cmap='magma', vmin=0.0, vmax=1.0)
        ax.set_ylabel(f"Layers {name1}", fontsize=AXES_FONT_SIZE)
        ax.set_xlabel(f"Layers {name2}", fontsize=AXES_FONT_SIZE)

        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_yticks([0, 3, 6, 9, 12])

        add_colorbar(im)
        plt.tight_layout()

        # plt.show()
        save_path = save_path.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig("{}.png".format(save_path))
        np.save("{}.npy".format(save_path), hsic_matrix)
        print("save figures to {}.png".format(save_path))
