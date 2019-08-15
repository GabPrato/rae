# Recursive Autoencoder
[Towards Lossless Encoding of Sentences](https://arxiv.org/abs/1906.01659)

## Requirements
* Python 3.7
* Pytorch 1.x
* (For optional dataset generation) h5py

## Training Example
```bash
python train.py
```
Dataset not included, but `dataset_generator.py` can be used to generate a hdf5 dataset file from a text file of tokenized sentences, one per line.

## Pretrained Model
Embedding size 2048:
```python
model.load_state_dict(torch.load('rae2048.pt'))
```
