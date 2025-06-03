# allegro-pol

`allegro-pol` is an extension package of the `nequip` framework that adapts the Allegro architecture (another `nequip` extension package) for the prediction of the electric response of materials (polarization, Born charges, polarizability) in addition to energy and forces within a single ML model.
The ideas are described in ["Unified differentiable learning of electric response"](https://www.nature.com/articles/s41467-025-59304-1).

## Installation

This installation requires `nequip>=0.10.1` and `nequip-allegro>=0.6.0`, which will automatically be installed with `allegro-pol`.

It is strongly recommended to create a fresh virtual environment. For example,
```
conda create -n allegro-pol python=3.11
conda activate allegro-pol
```

It may be advisable to install an older version of PyTorch (e.g. PyTorch 1.11).

Then, install `allegro-pol`, which will install all essential dependencies.
```
git clone https://github.com/mir-group/allegro-pol.git
cd allegro-pol
pip install -e .
```

Users may wish to install additional dependencies such as the Weights and Biases package for logging.
```
pip install wandb
```

## Example

BaTiO3 data and an associated config file are provided for training, which is all based on the `nequip` framework with minor extensions.
The data is located at `data/BaTiO3.xyz` and the example config is located at `configs/BaTiO3.yaml`.
```
nequip-train configs/BaTiO3.yaml
```

## Pre- and Post-Processing Scripts

Pre- and post-processing scripts can be found in `scripts`, along with a tutorial on how to use them.

## Cite

If you use this code in your own work, please cite [our work](https://www.nature.com/articles/s41467-025-59304-1):
```
@article{falletta2025unified,
  title={Unified differentiable learning of electric response},
  author={Falletta, Stefano and Cepellotti, Andrea and Johansson, Anders and Tan, Chuin Wei and Descoteaux, Marc L and Musaelian, Albert and Owen, Cameron J and Kozinsky, Boris},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={4031},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

Also consider citing:

 1. The [original Allegro paper](https://www.nature.com/articles/s41467-023-36329-y)
```
@article{musaelian2023learning,
  title={Learning local equivariant representations for large-scale atomistic dynamics},
  author={Musaelian, Albert and Batzner, Simon and Johansson, Anders and Sun, Lixin and Owen, Cameron J and Kornbluth, Mordechai and Kozinsky, Boris},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={579},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

 2. The [original NequIP paper](https://www.nature.com/articles/s41467-022-29939-5)
```
@article{batzner20223,
  title={E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

 3. The `e3nn` equivariant neural network package used by NequIP, through its [preprint](https://arxiv.org/abs/2207.09453) and/or [code](https://github.com/e3nn/e3nn)
