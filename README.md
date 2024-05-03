# LaB-GATr
[Disclaimer:] this repository is currently under construction and will be frequently updated.

## Installation
We recommend creating a new Anaconda environment:
```
conda create --name lab-gatr python=3.10
conda activate lab-gatr
```
Next, install PyTorch and xFormers depending on your system. In our case, this was
```
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121
```
Additonally, we need Pytorch Geometric (currently only v2.4.0) and some dependencies
```
pip install torch_geometric==2.4.0
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
You can now install `lab_gatr` (which also installs [`gatr`](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)) via
```
pip install .
```
from within this repository.

To run `main.py` we also need some util:
pip install h5py prettytable meshio

## Getting started
LaB-GATr requires two things: a point cloud pooling transform for the tokenisation (patching) and a geometric algebra interface to embed your data in $\mathbf{G}(3, 0, 1)$. In the following we provide a minimal working example.

Let us first create a dummy mesh: n positions and orientations (e.g. surface normal) and an arbitrary scalar feature (e.g. geodesic distance).
```
import torch

n = 10000

pos, orientation = torch.rand((n, 3)), torch.rand((n, 3))
scalar_feature = torch.rand(n)
```
We need to compute auxiliary tensors that will be used during tokenisation (patching). We use Pytorch Geometric.
```
from lab_gatr import PointCloudPoolingScales
import torch_geometric as pyg

transform = PointCloudPoolingScales(rel_sampling_ratios=(0.2,), interp_simplex='triangle')
data = transform(pyg.data.Data(pos=pos, orientation=orientation, scalar_feature=scalar_feature))
```
Next, we define the embedding of our data in $\mathbf{G}(3, 0, 1)$. This means setting the number of input and output channels plus some logic that wraps around the model. We package this interface as Python class for convenience.
```
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
```
from lab_gatr import LaBGATr

model = LaBGATr(GeometricAlgebraInterface, d_model=8, num_blocks=10, num_attn_heads=4, use_class_token=False)
output = model(data)
```
Setting `use_class_token=True` will result in mesh-level instead of vertex-level output.
