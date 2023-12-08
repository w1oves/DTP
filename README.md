# DTP (Disentangle then Parse)

This repository implements a PyTorch re-implementation of the research paper: ["Disentangle then Parse: Night-time Semantic Segmentation with Illumination Disentanglement"](https://arxiv.org/abs/).

![overview](https://github.com/w1oves/DTP/assets/54713447/d9725a14-7495-4740-ac0c-ed5597d45d20)

## Dataset

**Access the refined train and val dataset via [Google Drive](https://drive.google.com/file/d/1Ilj99NMAmkZIPQcVOd6cJebnKXjJ-Sit/view?usp=drive_link).**

## Environment Setup

Set up your environment with these steps:

```bash
conda create -n dtp python=3.10
conda activate dtp
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
mim install mmcv-full
pip install -v -e .
# Alternatively: python setup.py develop
```

## Preparation

### Download and Organize

1. Decompress the dataset and relocate it to `data/nightcity-fine`.
2. Download the checkpoint from [Google Drive](https://drive.google.com/file/d/1iAjmJKc6pww2Nm_Vz4fJqQ9EF5sjyEE0/view?usp=sharing) and place it in `checkpoints`.

Your directory structure should resemble:

```plaintext
.
├── checkpoints
│   ├── night
│   └── night+day
├── custom
├── custom-tools
│   ├── dist_test.sh
│   ├── dist_train.sh
│   ├── test.py
│   └── train.py
├── data
│   ├── cityscapes -> /data0/wzx/data/darkseg-data/cityscapes/
│   └── nightcity-fine
├── mmseg
├── readme.md
├── requirements.txt
├── setup.cfg
└── setup.py
```

## Testing

Execute tests using:

```bash
python custom-tools/test.py checkpoints/night/cfg.py checkpoints/night/night.pth --eval mIoU --aug-test
```

## Training

Start training with:

```bash
python custom-tools/train.py </path/to/your/config>
```

## Results

The table below summarizes our findings:

| logs                                            | train dataset                  | validation dataset | mIoU |
|-------------------------------------------------|--------------------------------|--------------------|------|
| checkpoints/night/eval_multi_scale_20230801_162237.json | nightcity-fine                 | nightcity-fine     | 64.2 |
| checkpoints/night+day/eval_multi_scale_20230809_170141.json | nightcity-fine + cityscapes    | nightcity-fine     | 64.9 |

# Acknowledgements
This dataset is refined based on the dataset of [NightCity](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html) by Xin Tan *et al.* and [NightLab](https://github.com/xdeng7/NightLab) by Xueqing Deng *et al.*.

This project is based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation.git).

The annotation process was completed using [LabelMe](https://github.com/wkentaro/labelme.git).

# Citation
If you find this code or data useful, please cite our paper
```
@InProceedings{Wei_2023_ICCV,
    author    = {Wei, Zhixiang and Chen, Lin and Tu, Tao and Ling, Pengyang and Chen, Huaian and Jin, Yi},
    title     = {Disentangle then Parse: Night-time Semantic Segmentation with Illumination Disentanglement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21593-21603}
}
```
