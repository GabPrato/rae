# Recursive Autoencoder
[Towards Lossless Encoding of Sentences](https://arxiv.org/abs/1906.01659)
```bash
python train_example.py
```

*rae2048.pt* is a pretrained model with embedding size 2048.
```python
model.load_state_dict(torch.load('rae2048.pt'))
```
