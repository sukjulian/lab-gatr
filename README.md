# :lab_coat: LaB-GATr :crocodile:
This repository contains the official implementation of ["LaB-GATr: geometric algebra transformers for large biomedical surface and volume meshes"](https://arxiv.org/abs/2403.07536) (MICCAI 2024) and ["Geometric algebra transformers for large 3D meshes via cross-attention"](https://openreview.net/forum?id=T2bBUlaJTA) (GRaM workshop @ ICML 2024).

## Installation
We recommend creating a new Anaconda environment (tested on Python 3.11):
```shell
conda create --name lab-gatr python=3.11
conda activate lab-gatr
```
Next, install PyTorch and xFormers (tested on the following versions) depending on your system. In our case, this was
```shell
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
```
Additonally, we need PyTorch Geometric and some dependencies (tested on the following versions)
```shell
pip install torch_geometric==2.4.0
pip install torch_scatter torch_cluster --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
You can now install [`gatr`](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer):
```shell
pip install git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git
```
and then `lab_gatr` via
```shell
pip install .
```
from within this repository. If you experience performance issues with current versions of some of these packages, consider resetting to the above versions.

## Getting started
LaB-GATr requires two things: a point cloud pooling transform for the tokenisation (patching) and a geometric algebra interface to embed your data in $\mathbf{G}(3, 0, 1)$. In the following we provide a minimal working example.

Let us first create a dummy mesh: n positions and orientations (e.g. surface normal) and an arbitrary scalar feature (e.g. geodesic distance).
```python
import torch

n = 10000

pos, orientation = torch.rand((n, 3)), torch.rand((n, 3))
scalar_feature = torch.rand(n)
```
We need to compute auxiliary tensors that will be used during tokenisation (patching). We use Pytorch Geometric.
```python
from lab_gatr import PointCloudPoolingScales
import torch_geometric as pyg

transform = PointCloudPoolingScales(rel_sampling_ratios=(0.2,), interp_simplex='triangle')
data = transform(pyg.data.Data(pos=pos, orientation=orientation, scalar_feature=scalar_feature))
```
Next, we define the embedding of our data in $\mathbf{G}(3, 0, 1)$. This means setting the number of input and output channels plus some logic that wraps around the model. We package this interface as Python class for convenience.
```python
from gatr.interface import embed_oriented_plane, extract_translation

class GeometricAlgebraInterface:
    num_input_channels = num_output_channels = 1
    num_input_scalars = num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data):

        multivectors = embed_oriented_plane(normal=data.orientation, position=data.pos).view(-1, 1, 16)
        scalars = data.scalar_feature.view(-1, 1)

        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        output = extract_translation(multivectors).squeeze()

        return output
```
That's it! With the interface class we just defined, we can create the model and run inference.
```python
from lab_gatr import LaBGATr

model = LaBGATr(GeometricAlgebraInterface, d_model=8, num_blocks=10, num_attn_heads=4, use_class_token=False)
output = model(data)
```
Setting `use_class_token=True` will result in mesh-level instead of vertex-level output.

## New features
Besides tokenisation via message passing, `lab_gatr` now also supports cross-attention for squence reduction. You can switch between the two by setting `pooling_mode='message_passing'` or `pooling_mode='cross_attention'` (default).

## Citation
If you use LaB-GATr in your research, please cite either (or both):
```
@inproceedings{LaBGATrMICCAI,
  author={Julian Suk and Baris Imre and Jelmer M. Wolterink},
  title={{LaB-GATr}: geometric algebra transformers for large biomedical surface and volume meshes},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024},
  publisher={Springer Nature Switzerland},
  address={Cham},
  pages={185--195},
  isbn={978-3-031-72390-2}
}

@inproceedings{LaBGATrGRaM,
  title={Geometric algebra transformers for large {3D} meshes via cross-attention},
  author={Julian Suk and Pim de Haan and Baris Imre and Jelmer M. Wolterink},
  booktitle={ICML Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)},
  year={2024},
  url={https://openreview.net/forum?id=T2bBUlaJTA}
}
```
