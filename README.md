# DeS3_Deshadow (AAAI'2024)

## Introduction
This is an implementation of DeS3: Attention-driven Self and Soft Shadow Removal using ViT Similarity and Color Convergence

```
git clone https://github.com/jinyeying/DeS3_Deshadow.git
cd DeS3_Deshadow/
```

## 1. Datasets
1. SRD [Train](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view)|[BaiduPan](https://pan.baidu.com/s/1mj3BoRQ), [Test](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view).
[Shadow Masks](https://github.com/vinthony/ghost-free-shadow-removal)

2. AISTD|ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID) 

3. LRSS: Soft Shadow Dataset [[link]](http://visual.cs.ucl.ac.uk/pubs/softshadows/)<br>
   The LRSS dataset contains 134 shadow images (62 pairs of shadow and shadow-free images). <br>
   We use 34 pairs for testing and 100 shadow images for training. For shadow-free training images, 28 from LRSS and 72 randomly selected from the USR dataset.<br>
   |[[Dropbox]](https://www.dropbox.com/scl/fo/3dt75e23riozwa6uczeqd/ABNkIZKaP8jFarfNrUUjpVg?rlkey=eyfjn7dhd9pbz6rh247ylbt0c&st=01lh80r8&dl=0)|[[BaiduPan (code:t9c7)]](https://pan.baidu.com/s/1c_VsDVC92WnvI92v8cldsg?pwd=t9c7)|
   | :-----------: | :-----------: |
  
5. USR: Unpaired Shadow Removal Dataset [[link]](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)

6. UCF, UIUC: Self Shadow [[link]](https://drive.google.com/file/d/1jyzJm13VbvXGocwmywsJ51yqWxAwt_pP/view)

### 2. SRD Dataset Results:
|[[Dropbox]](https://www.dropbox.com/scl/fo/04qdaxrapog8vvikh24d5/h?rlkey=u3e4trwim1im4c2yvc8ig1duq&dl=0) | [[BaiduPan(code:blk7)]](https://pan.baidu.com/s/1b-Elx5a9NHL5E0z_aHoydw?pwd=blk7)|
| :-----------: | :-----------: |

### SRD Dataset Evaluation
1. set the paths of the shadow removal result and the dataset in `evaluation/demo_SRD_RMSE.m` and then run it.
```
demo_SRD_RMSE.m
```
Get the RMSE from Table 1 in the main paper on the SRD (size: 256x256):

| Method   |Training| Shadow   | Non-Shadow |    ALL    |
|----------|--------|----------|------------|-----------|
| **DeS3** | Paired | **5.88** | **2.83** | **3.72** |

2. set the paths of the shadow removal result and the dataset in `evaluation/evaluate_SRD_PSNR_SSIM.m` and then run it.
```
evaluate_SRD_PSNR_SSIM.m
```
Get the PSNR & SSIM from Table 1 in the main paper on the SRD (size: 256x256):

|          |        |   PSNR   |    PSNR    |     PSNR  |   SSIM   |    SSIM    |    SSIM   |
|----------|--------|----------|------------|-----------|----------|------------|-----------| 
| Method   |Training| Shadow   | Non-Shadow |    ALL    | Shadow   | Non-Shadow |     ALL   |
| **DeS3** | Paired | **37.45**| **38.12**  | **34.11** | **0.984**| **0.988**  | **0.968** |

### 3. AISTD Dataset Results:
|[[Dropbox]](https://www.dropbox.com/scl/fo/72cvwfs78tvxy8myj3r3j/AOfsrY7CuexxfdYr0_qJPEY?rlkey=58dtz96rfrbn0t9oaff7phpv6&st=jzslheyv&dl=0) | [[BaiduPan(code:blk7)]](https://pan.baidu.com/s/1b-Elx5a9NHL5E0z_aHoydw?pwd=blk7)|
| :-----------: | :-----------: |

### AISTD Dataset Train, Test 
1. modify the path in https://github.com/jinyeying/DeS3_Deshadow/blob/c294476d562b65c8acbf2be8bc0986ebeab00c63/datasets/aistdshadow.py#L30 https://github.com/jinyeying/DeS3_Deshadow/blob/c294476d562b65c8acbf2be8bc0986ebeab00c63/configs/AISTDshadow.yml#L6
2. download the AISTD checkpoint [[Dropbox]](https://www.dropbox.com/s/q2qgb2e02h48q00/AISTDShadow_ddpm.pth.tar?dl=0) | [[BaiduPan(code:aistd)]](https://pan.baidu.com/s/1PDQXHfE7XUTo_jVpnFiReQ?pwd=aist) 

```
CUDA_VISIBLE_DEVICES=1,2 python train_aistd.py --config 'AISTDshadow.yml' --resume '/home1/yeying/DeS3_Deshadow/ckpts/AISTDShadow_ddpm.pth.tar'
```
```
CUDA_VISIBLE_DEVICES=1 python eval_aistd.py --config 'AISTDshadow.yml' --resume '/home1/yeying/DeS3_Deshadow/ckpts/AISTDShadow_ddpm.pth.tar'
```

### AISTD Dataset Evaluation
1. set the paths of the shadow removal result and the dataset in `evaluation/demo_AISTD_RMSE.m` and then run it.
```
demo_AISTD_RMSE.m
```
Get the RMSE on the AISTD (size: 256x256):

| Method   |Training| Shadow   | Non-Shadow |    ALL    |
|----------|--------|----------|------------|-----------|
| **DeS3** | Paired | **6.56** | **3.40** | **3.94** |

2. set the paths of the shadow removal result and the dataset in `evaluation/evaluate_AISTD_PSNR_SSIM.m` and then run it.
```
evaluate_AISTD_PSNR_SSIM.m
```
Get the PSNR & SSIM on the ISTD (size: 256x256):

|          |        |   PSNR   |    PSNR    |     PSNR  |   SSIM   |    SSIM    |    SSIM   |
|----------|--------|----------|------------|-----------|----------|------------|-----------| 
| Method   |Training| Shadow   | Non-Shadow |    ALL    | Shadow   | Non-Shadow |     ALL   |
| **DeS3** | Paired | **36.49**| **34.70**  | **31.38** | **0.989**| **0.972**  | **0.958** |


## Acknowledgments
Code is implemented based [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion), we would like to thank them.

## License
The code and models in this repository are licensed under the MIT License for academic and other non-commercial uses.<br>
For commercial use of the code and models, separate commercial licensing is available. Please contact:
- Jonathan Tan (jonathan_tano@nus.edu.sg)

### Citations
If this work is useful for your research, please cite our paper. 
```BibTeX
@inproceedings{jin2024des3,
  title={DeS3: Adaptive Attention-Driven Self and Soft Shadow Removal Using ViT Similarity},
  author={Jin, Yeying and Ye, Wei and Yang, Wenhan and Yuan, Yuan and Tan, Robby T},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={3},
  pages={2634--2642},
  year={2024}
}
```
