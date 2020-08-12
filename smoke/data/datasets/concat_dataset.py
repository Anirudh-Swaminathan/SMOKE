# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

import logging


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """

    def get_idxs(self, idx):
        l = logging.getLogger(__name__)
        l.info("IN ConcatDataset get_idxs({})".format(idx))
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        l = logging.getLogger(__name__)
        l.info("IN ConcatDataset get_img_info({})".format(idx))
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)
