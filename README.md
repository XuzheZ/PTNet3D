# Pyramid Transformer Net-3D (PTNet3D)
Public code for 3D version of Pyramid Transformer Network (PTNet3D). Our PTNet achieve consistently better performance in high-resolution and longitudinal infant brain MRI synthesis than previous SOTA methods. 

Example on synthesizing 3-month-old infant T2w conditioned on T1w:
![3Months](3mres-10.png)

## Paper link:
[IEEE-TMI (early access)](https://ieeexplore.ieee.org/document/9774943)

## Usage: 

### Training (quick start): 
    python train.py --name YourProjName --checkpoints_dir YourModelPath --dataroot YourDatasetPath 
You can change more hyperparameters in the option scripts. For a complete set of options, please check option scripts in ./options
### Data structure:
    ./opt.dataroot 
        - train_A # your source domain scans
        - train_B # your target domain scans
        - test_A # will be used for inference
the model is trained to convert modality in train_A to modality in train_B, please make sure all scans are well-coregistered and artifacts-free. Scans in train_A and train_B shall have the same file name.

### Inference: 
    python test.py --name YourProjName --checkpoints_dir YourModelPath --dataroot YourDatasetPath --whichmodel YourModelName
the model specified by opt.whichmodel will be used to convert the scans in /opt.dataroot/test_A. opt.whichmodel should be the full name of stored checkpoint. 

## Training tips:
The experiments in our paper were conducted using 2D slice or 3D block (64x64x64). Using the default configuration, a GPU with 24GB VRAM should be able to hold a mini-batch up to 4 (64x64x64 blocks).

In 2D experiments, we just resliced volumetric scan by taking the axial slice. It resulted in **~200 slices per scan**. 

In 3D experiments, we first use sliding window sampling to sample 3D blocks from the whole brain (non-zero region) with overlapping. By doing that, we generated over **200 ROIs per scans**. Therefore, in training, we trained the model for 10 epochs on dHCP and 6 epochs on BCP because we have ~ 50,000 ROIs on each datasets. 

In this repo, to simplify the sampling, we choose **random sampling** in non-zero region of the volumetric scan instead of excessive overlapping sampling (details in ./data/aligned_dataset.py). Therefore, you might need use **a larger number epochs** for training. For instance, if we re-train the model on dHCP dataset, we should use **2,000 epochs** instead of 10 epochs for training because in each epoch, there are only 291 randomly-sampled ROIs. 

## Citation: 
If you use our code in your research, please cite our paper: [PTNet3D: A 3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers](https://ieeexplore.ieee.org/document/9774943)
```
@ARTICLE{9774943,
  author={Zhang, Xuzhe and He, Xinzi and Guo, Jia and Ettehadi, Nabil and Aw, Natalie and Semanek, David and Posner, Jonathan and Laine, Andrew and Wang, Yun},
  journal={IEEE Transactions on Medical Imaging},
  title={PTNet3D: A 3D High-Resolution Longitudinal Infant Brain MRI Synthesizer Based on Transformers},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3174827}}
```
