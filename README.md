```
**Optimization-Inspired Dilated Deep Network for Compressive Sensing of Color Images**
```
Image compressive sensing reconstruction methods are mainly divided into traditional optimization algorithms and deep learning methods, and the latter is more popular because of its fast computing speed and good reconstruction quality. However, most of the current deep learning-based reconstruction methods are designed for grayscale images, and when these methods are used for color image compression and perception, they usually use the same sampling matrix for channel-by-channel sampling and reconstruction without considering the strong correlation between channels, so the reconstruction results are not ideal. Few deep learning models designed for color image compression and perception lack interpretability and have performance bottlenecks because they are based on purely data-driven design. In this paper, we design a deep network OIDDN based on heuristic optimization for color image compression perception, using three channels independently sampled and jointly trained; the iterative solution steps of FISTA algorithm are strictly corresponded to each stage of the reconstructed network module during reconstruction; in addition, the model adds an adaptive expansion convolution module to increase the multi-scale dynamic perception capability of the network. The experimental results show that the proposed method has significantly improved the reconstruction performance in comparison with existing deep learning models for image compression perception in public datasets.
```
Software Dependencies
```
python 3.8.10<br>
pytorch 1.9.0<br>
CUDA 10.2<br>
scipy 1.6.2<br>
numpy 1.20.2<br>
matplotlib 3.3.4<br>
h5py 2.10.0<br>
The code of this project was developed and tested in windows 10 environment, other environments were not tested.
```
Generate sample data
```
Raw training image data comes from [BSD-500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)，Put it in the data directory and use python to execute CSDataSetBuilderWithChannelFirst.py，Training data sets can be generated。
```
Training the OIDN model
```
```
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 1
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 5
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 10
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 20
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 25
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 30
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 40
python Train_OIDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 50
```
Pre-training models are available in the model directory with 8 sampling ratios of 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4 and 0.5, which can be downloaded for direct testing and use.
```
OIDN Model Testing
```
Put the test set directory into the data folder, support bmp,png and jpg formats, pass in different cs_ratio and test_name to test different sample rates and test image sets
```
python Test_OIDN.py --layer_num 9 --cs_ratio 1 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 5 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 10 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 20 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 25 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 30 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 40 --test_name Set5
python Test_OIDN.py --layer_num 9 --cs_ratio 50 --test_name Set5
```
```
Training the OIDDN model
```
```
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 1
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 5
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 10
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 20
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 25
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 30
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 40
python Train_OIDDN.py --start_epoch 0 --end_epoch 200 --layer_num 9 --learning_rate 1e-4 --cs_ratio 50
```
```
OIDDN model testing
```
Pass in different cs_ratio and test_name to test different sample rates and test image sets
```
python Test_OIDDN.py --layer_num 9 --cs_ratio 1 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 5 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 10 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 20 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 25 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 30 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 40 --test_name Set5
python Test_OIDDN.py --layer_num 9 --cs_ratio 50 --test_name Set5
```
```
Statement
```
This project is my master's research results, supervised by [Jian Zhang](https://github.com/jianzhangcs), now we open the source code. And inevitably there will be mistakes, welcome to correct by email.