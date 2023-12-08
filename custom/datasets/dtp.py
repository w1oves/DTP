from mmseg.datasets.builder import DATASETS
from mmseg.datasets import build_dataset
import torch
@DATASETS.register_module()
class DTPDataset(object):
    def __init__(self,datasetA,datasetB) -> None:
        self.datasetA=build_dataset(datasetA)
        self.datasetB=build_dataset(datasetB)
        self.CLASSES=self.datasetA.CLASSES
        self.PALETTE=self.datasetA.PALETTE
    def __len__(self):
        return len(self.datasetA)
    def __getitem__(self,index):
        resultA=self.datasetA[index]
        resultB=self.datasetB[torch.randint(0,len(self.datasetB),[]).item()]
        return dict(
            imgA=resultA['img'],
            gtA=resultA['gt_semantic_seg'],
            metasA=resultA['img_metas'],
            imgB=resultB['img'],
            gtB=resultB['gt_semantic_seg'],
            metasB=resultB['img_metas'],
        )