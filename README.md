# MNIST_Classifier_CNN_vs_Transformer
From-scratch PyTorch implementations of a multi-branch CNN and a Vision Transformer for MNIST — 99.18% and 99.29% test accuracy, with pretrained checkpoints included.

| Model | Notebook | Architecture | Test Accuracy | Parameters |
|---|---|---|---|---|
| Multi-Scale CNN | [`MNIST_CNN.ipynb`](MNIST_CNN.ipynb) | Two parallel conv branches (2x2 / 4x4 kernels) → concat → pooling → FC head | **99.18%** | 133,578 |
| Vision Transformer | [`MNIST_Transformer.ipynb`](MNIST_Transformer.ipynb) | Patch embedding (2x2 patches) + 4-layer Transformer encoder + `[CLS]` token | **99.29%** | 557,450 |

## Repository structure

```
.
├── MNIST_CNN.ipynb
├── MNIST_Transformer.ipynb
├── models/
│   ├── mnist_cnn_model.pth
│   └── mnist_transformer_model.pth
├── requirements.txt
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook
```

Each notebook works in two modes, controlled by a single flag in its "Configuration" cell:

- **Evaluate the pretrained model** (default, `TRAIN_FROM_SCRATCH = False`): loads the checkpoint from `models/` and reproduces the reported test accuracy directly.
- **Train from scratch** (`TRAIN_FROM_SCRATCH = True`): retrains the model with the same hyperparameters used to produce the released checkpoints.

Both notebooks download MNIST automatically via `torchvision` on first run.

## Results

### Multi-Scale CNN
- Test accuracy: **99.18%**
- Best validation accuracy during training: 99.23% (early stopped at epoch 16/50)
- Loss: cross-entropy with label smoothing (0.05)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)

### Vision Transformer
- Test accuracy: **99.29%** (test loss: 0.5180)
- Loss: cross-entropy with label smoothing (0.1)
- Optimizer: AdamW (lr=6e-4, weight_decay=1e-2)
- Gradient clipping: max norm 1.0

## License

No license has been set yet — add one (e.g. MIT) if you want to explicitly allow reuse.
