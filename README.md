# DeS3_Deshadow (AAAI'2024)

## Introduction
This is an implementation of DeS3: Attention-driven Self and Soft Shadow Removal using ViT Similarity and Color Convergence

### 1. SRD Dataset Results:
[Dropbox](https://www.dropbox.com/scl/fo/04qdaxrapog8vvikh24d5/h?rlkey=u3e4trwim1im4c2yvc8ig1duq&dl=0) |
[BaiduPan](https://pan.baidu.com/s/1b-Elx5a9NHL5E0z_aHoydw?pwd=blk7) code:blk7


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

|    || PSNR   | PSNR |     PSNR   | SSIM   | SSIM |     SSIM  |
|----------|--------|----------|------------|-----------|----------|------------|-----------| 
| Method   |Training| Shadow   | Non-Shadow |    ALL    | Shadow   | Non-Shadow |     ALL   |
| **DeS3** | Paired | **37.45**| **38.12** | **34.11**  |**0.984** | **0.988**  | **0.968** |

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
