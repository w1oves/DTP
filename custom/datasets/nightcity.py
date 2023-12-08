# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets import CityscapesDataset


@DATASETS.register_module()
class NightcityDataset(CityscapesDataset):
    """NightCityDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".png", seg_map_suffix="_trainIds.png", **kwargs)
