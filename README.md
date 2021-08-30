```
OIDDN
```
图像压缩感知重建方法主要分为传统优化算法和深度学习方法，后者因运算速度快、重建质量好更受人们青睐。然而，当前基于深度学习的重建方法大多面向灰度图像进行设计，此类方法在用于彩色图像压缩感知时，通常采用相同的采样矩阵逐通道采样及重建，未考虑通道间的强相关性因此重建结果不理想。极少数针对彩色图像压缩感知设计的深度学习模型因为基于纯数据驱动设计故而缺乏可解释性，并存在性能瓶颈。本文设计了一种应用于彩色图像压缩感知的基于启发优化的深度网络OIDDN，采用三通道独立采样并联合训练；重建时将FISTA算法迭代求解步骤严格对应到每一阶段的重建网络模块；另外模型加入了自适应扩张卷积模块，进而增加网络多尺度动态感知能力。实验结果表明，本文所提方法相对现有图像压缩感知深度学习模型在公开数据集对比中重建性能有了大幅提升。
```
环境依赖
```
pytorch 1.9.0
scipy 1.6.2
numpy 1.20.2
python 3.8.10
matplotlib 3.3.4
h5py 2.10.0
```
生成样本数据
```
原始训练图像数据来自 [BSD-500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)，将其放入data目录，使用python 执行 CSDataSetBuilderWithChannelFirst.py，可生成训练数据集。
```
训练模型
```
```
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 1
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 5
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 10
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 20
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 25
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 30
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 40
python Train.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 50
```
```
模型测试
```
传入 cs_ratio 和 test_name 用来测试不同的采样率和测试图片集
```
python Test.py --layer_num 9 --cs_ratio 1 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 5 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 10 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 20 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 25 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 30 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 40 --test_name Set5
python Test.py --layer_num 9 --cs_ratio 50 --test_name Set5
```