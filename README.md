# Uni-Adapter
This repository is the official implementation of the AAAI 2026 paper ["Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models"](http://arxiv.org/link).




## Evaluation

### MODE-DOTA(ours)
```sh
python main_test-time.py --use-mode-dota --mode-M 8 --res-learning \ #mode dota parameters
   --dota-sigma 0.0001 --dota-eta 0.1 --dota-rho 0.02  # same as dota parameters

```
### Parameters

- `--use-mode-dota`: Enable MODE-DOTA test-time adaptation. (OURS)
- `--mode-M`: Number of modes per class for MODE DOTA.
- `--res-learning`: Enable residual learning for MODE DOTA
- `--dota-epsilon`: DOTA & MODE DOTA hyperparameter epsilon (for covariance regularization). -> same for dota and mode dota(ours)
- `--dota-sigma`: DOTA & MODE DOTA hyperparameter sigma (for initial covariance). -> same for dota and mode dota(ours)
- `--dota-eta`: DOTA & MODE DOTA hyperparameter eta (for fusion weight scaling). -> same for dota and mode dota(ours)
- `--dota-rho`: DOTA & MODE DOTA hyperparameter rho (for fusion weight initial value). -> same for dota and mode dota(ours)
- `--use-dota`: Enable DOTA test-time adaptation.




### DOTA

```sh
python main_test-time.py --use-dota \
   --dota-sigma 0.0001 --dota-eta 0.1 --dota-rho 0.02


```

### Parameters

- `--video`: Path to input video file (required)
- `--ckpt`: Path to StreamVGGT checkpoint (default: automatic download from HuggingFace)
- `--out_dir`: Output directory for results (default: "output_streamvggt")
- `--fps_interval`: Extract 1 frame every N seconds (default: 2.5)
- `--conf_thres`: Confidence threshold for 3D visualization (default: 3.0)
- `--show_cam`: Show camera poses in 3D visualization
- `--mask_black_bg`: Mask black background pixels
- `--mask_white_bg`: Mask white background pixels  
- `--mask_sky`: Apply sky segmentation mask


### Pre-trained Weights

We use the following models as baselines. Their pre-trained weights can be obtained from the respective public repositories:

* [ULIP-2](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip-2)  
* [OpenShape](https://github.com/Colin97/OpenShape_code/)  
* [Uni3D](https://github.com/baaivision/Uni3D)  

**Details on the models and text encoders used:**

1. **ULIP-2**  
   - Model: Use the same [text encoder](https://huggingface.co/datasets/auniquesun/Point-PRC/tree/main/pretrained-weights/ulip/image-text-encoder).  

2. **OpenShape**  
   - Model: [pointbert-vitg14-rgb](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main)  
   - Text Encoder: [CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)  

3. **Uni3D**  
   - Model: [uni3d-L](https://huggingface.co/BAAI/Uni3D/tree/main/modelzoo/uni3d-L)  
   - Text Encoder: [eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k](https://huggingface.co/timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k)



### Datasets
1. The folder structure of used datasets should be organized as follows.
```sh
    
  |----data   
      |----modelnet40_c
      |----shapenet_c
      |----scanobjnn_c
      |----omniobject3d
          |----1024
      |----objaverse_lvis


    ...
```

### Evaluation on _ModelNet-C_

For ModelNet40-C, we used the corrupted data from [Modelnet40-C](https://github.com/jiachens/ModelNet40-C) repository, using the maximum corruption severity level (5) for all experiments



## Acknowledgement
Our implementation is partially inspired by the following projects, thanks to their great work.

1. [Point-Cache](https://github.com/auniquesun/Point-Cache)
2. [ULIP](https://github.com/salesforce/ULIP)
3. [OpenShape](https://github.com/ZrrSkywalker/PointCLIP)
4. [Uni3D](https://github.com/yangyangyang127/PointCLIP_V2)
5. [TDA](https://github.com/kdiAAA/TDA)

## Contact
If you have any question about our work, please search related issues or create a new one in this repository.
