# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from .misc import DPI, BASE_FONT_SIZE, FONT_NAME, AXES_FONT_SIZE


class _Analyzer(object):
    def __init__(self, num_tokens=196, num_heads=12):
        self._samples_num = 0
        self._avg_entropy = np.zeros(num_heads)
        n = np.sqrt(num_tokens)
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        xx, yy = np.meshgrid(x.flatten(), y.flatten())
        self._dist_matrix = np.sqrt(np.square(xx-xx.T) + np.square(yy-yy.T)) * 14
        self._avg_distance = np.zeros(num_heads)

    def append(self, attn):
        # b, num_heads, num_tokens, num_tokens -> b, num_heads, num_tokens
        entropy = self._get_entropy(attn)
        mean_entropy = entropy.mean(dim=(0, 2)).numpy()
        self._avg_entropy = (self._avg_entropy*self._samples_num + mean_entropy) / (self._samples_num+1)
        distance = self._get_dist(attn)
        mean_distance = distance.mean(dim=(0, 2)).numpy()
        self._avg_distance = (self._avg_distance*self._samples_num + mean_distance) / (self._samples_num+1)
        self._samples_num += 1

    def get(self):
        return self._avg_entropy, self._avg_distance

    def _get_entropy(self, attn):
        # b, num_heads, num_tokens, num_tokens
        attn = attn.softmax(dim=-1)
        entropy = (-attn*(attn+1e-6).log()).sum(dim=-1)
        return entropy

    def _get_dist(self, attn):
        attn = attn.softmax(dim=-1)
        dist = (attn * self._dist_matrix[None, None, :, :]).sum(axis=-1)
        return dist


class AttnAnalyzer(object):
    def __init__(self, nums, num_heads=12):
        self.analyzer_group = [_Analyzer(num_heads=num_heads) for _ in range(nums)]

    def append(self, attns):
        for attn, analyzer in zip(attns, self.analyzer_group):
            analyzer.append(attn[:, :, 1:, 1:].cpu())
    
    def get(self):
        results = [analyzer.get() for analyzer in self.analyzer_group]
        entropy = [r[0] for r in results]
        distance = [r[1] for r in results]
        return entropy, distance

    def plot_attn(self, save_path, title=""):
        results = self.get()
        plt.rcParams['font.size'] = BASE_FONT_SIZE
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = [FONT_NAME]
        save_path = save_path.replace(" ", "_").replace("(", "").replace(")", "")
        entropy = np.array(results[0])
        depth, num_head = entropy.shape
        fig, ax1 = plt.subplots(dpi=DPI)
        for i in range(depth):
            ax1.scatter(np.ones(num_head)*(i+1), entropy[i], alpha=0.6)
        ax1.plot(range(1, depth+1), entropy.mean(axis=1), color='black', marker='*', markersize=12)
        ax1.set_xlabel("Depth", fontsize=AXES_FONT_SIZE)
        ax1.set_ylabel("Attention Entropy", fontsize=AXES_FONT_SIZE)
        ax1.set_xticks([1, 3, 6, 9, 12])
        ax1.set_ylim(0, 6)
        fig.tight_layout()
        plt.savefig("{}.png".format(save_path.replace("attn_", "attne_")))
        print("save figures to {}.png".format(save_path.replace("attn_", "attne_")))
        
        fig, ax2 = plt.subplots(dpi=DPI)
        distance = np.array(results[1])
        for i in range(depth):
            ax2.scatter(np.ones(num_head)*(i+1), distance[i], alpha=0.6)
        ax2.plot(range(1, depth+1), distance.mean(axis=1), color='black', marker='*', markersize=12)
        ax2.set_xlabel("Depth", fontsize=AXES_FONT_SIZE)
        ax2.set_ylabel("Mean Distance", fontsize=AXES_FONT_SIZE)
        ax2.set_xticks([1, 3, 6, 9, 12])
        ax2.set_ylim(0, 150)
        fig.tight_layout()

        plt.savefig("{}.png".format(save_path.replace("attn_", "attnd_")))
        print("save figures to {}.png".format(save_path.replace("attn_", "attnd_")))

        np.save("{}.npy".format(save_path), [entropy, distance])

    def plot_box_attn(self, save_path, title=""):
        results = self.get()
        plt.rcParams['font.size'] = BASE_FONT_SIZE
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = [FONT_NAME]
        save_path = save_path.replace(" ", "_").replace("(", "").replace(")", "")
        entropy = np.array(results[0])
        depth, num_heads = entropy.shape
        fig, ax1 = plt.subplots(dpi=DPI)
        col = '#1f77b480'
        ax1.boxplot(
            entropy.T, labels=list(map(str, range(1, depth+1))), patch_artist=True, 
            boxprops={"facecolor": col, "edgecolor": col[:-2]},
            flierprops={"marker": 'o', "markerfacecolor": col, "markeredgecolor": col[:-2]},
            medianprops={"color": col[:-2]},
            whiskerprops={"color": col[:-2]},
            capprops={"color": col[:-2]}
        )
        ax1.set_xlabel("Depth", fontsize=AXES_FONT_SIZE)
        ax1.set_ylabel("Average Entropy", fontsize=AXES_FONT_SIZE)
        # ax1.set_xticks(range(1, 12, 2))
        ax1.set_ylim(0, 6)
        fig.tight_layout()
        plt.savefig("{}.png".format(save_path.replace("attn_", "attne_")))
        print("save figures to {}.png".format(save_path.replace("attn_", "attne_")))
        
        fig, ax2 = plt.subplots(dpi=DPI)
        distance = np.array(results[1])
        ax2.boxplot(
            distance.T, labels=list(map(str, range(1, depth+1))), patch_artist=True, 
            boxprops={"facecolor": col, "edgecolor": col[:-2]},
            flierprops={"marker": 'o', "markerfacecolor": col, "markeredgecolor": col[:-2]},
            medianprops={"color": col[:-2]},
            whiskerprops={"color": col[:-2]},
            capprops={"color": col[:-2]}
        )
        ax2.set_xlabel("Depth", fontsize=AXES_FONT_SIZE)
        ax2.set_ylabel("Average Distance", fontsize=AXES_FONT_SIZE)
        # ax2.set_xticks(range(1, 12, 2))
        ax2.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        ax2.set_ylim(0, 135)
        fig.tight_layout()

        plt.savefig("{}.png".format(save_path.replace("attn_", "attnd_")))
        print("save figures to {}.png".format(save_path.replace("attn_", "attnd_")))

        np.save("{}.npy".format(save_path), [entropy, distance])


if __name__ == "__main__":
    analyzer = _Analyzer()
    attn_list = [torch.randn(2, 3, 10, 10) for _ in range(4)]
    for attn in attn_list:
        analyzer.append(attn)
    print(analyzer.get())

    attns = torch.cat(attn_list)
    analyzer2 = _Analyzer()
    analyzer2.append(attns)
    print(analyzer2.get())