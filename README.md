# Video Saliency model 

## ViNet

Implementation of paper (unofficial) - [ViNet: Pushing the limits of Visual Modality for Audio-Visual Saliency Prediction]  
github:(https://github.com/samyak0210/ViNet)

## Dataset
DHF1K  
github:(https://github.com/wenguanwang/DHF1K)

## Training
```shell
python main.py --model ViNet --num_epochs 50 --root ./dataset/DHF1K --path_backbone_pretrained ./S3D_kinetics400.pt --batch_size 8 --kl 

```


