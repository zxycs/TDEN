# Lightweight Image Super-Resolution with Tokenized Dynamic Embedding Network

## Environment
- Python 3.9
- PyTorch 2.0.1
  
## Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/TDEN_SRx4.yml --launcher pytorch
```
## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.
### Testing Commands
```bash
python basicsr/test.py -opt options/test/TDEN_SRx4.yml
```
