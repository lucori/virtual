# VIRTUAL

The VIRTUAL package implements a model to perform federated multi-task learning with with variational NNs.


## Getting Started

We recommend to setup Miniconda to create a python environment from the enviroment file environment.yml

```
conda env create -f environment.yml
source activate virtual
```

Additionally install pip packages in the same environment:

```
pip install -r requirements.txt
```

## General usage


To reproduce the experiments of the paper use the main.py file, giving a configuration file from the configuration folder as 

```
python main.py configurations/femnist_virtual.json 
```

Then track the experiment using tensorboard as 

```
tensorboard --logdir logs/femnist_virtual_*
```

Hyperparameters and relative metrics are tracked using the HPARAM API of tensorboard (see https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams).

Note that if you can not use any GPU, you have to specify "session": { "num_gpus": 0} in the config file.