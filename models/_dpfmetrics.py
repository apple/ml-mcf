import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import molecule_gen_metric


class _DPFMetrics:
    def __init__(self):
        return

    def metrics_molecule(self, sample_mols_list, gt_mols_list):
        return molecule_gen_metric(
            sample_mols_list, gt_mols_list, threshold=self.threshold
        )
