# PTF-EvoMDE 


## Requirements

base: cuda==11.8 

```
conda create -n PTF-EvoMDE python=3.8 
conda activate PTF-EvoMDE

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# pip
pip install cython
pip install mmcv==0.2.10
pip install tqdm
pip install einops
pip install fvcore
pip install timm==0.4.12

# setup
sh ./mmdet_build.sh

cd DCNv2_latest && python3 setup.py build develop
```

## Datasets

[NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html),[KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction),[synthetic data](http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/)

## Searching
soon

## Training
soon

## Evaluation

Evaluate the searched model on NYUv2:

```
sh scripts/test/nyu_test.sh
```

Evaluate the searched model on KITTI:
```
sh scripts/test/kitti_test.sh
```

Evaluate on synthetic data:
```
sh scripts/test/colon_test.sh
```

## Models
| Model | Params.| Abs.Rel. | Sqr.Rel | RMSE | RMSElog | a1 | a2 | a3| 
| :--- | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
|[NYUv2](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_nyu.ckpt) | 6.13 | 0.0952 | 0.0443 | 0.3310 | 0.1185 | 0.923 | 0.992 | 0.998  |
|[KITTI_Eigen](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_kittieigen.ckpt) | 6.26 | 0.0520 | 0.1482 | 2.0716 | 0.0780 | 0.975 | 0.997 | 0.999 |

| Model | Params.| Mean L1-error  | Mean Relative L1-error  | Mean RMSE |
| :--- | :---: | :---: | :---: |  :---: | 
|[Colon](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/models/model_nyu.ckpt) | 6.26 | 0.0952 | 0.0443 | 0.3310 | 