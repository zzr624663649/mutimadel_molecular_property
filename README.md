## Boosting the performance of molecular property prediction via graph-text alignment and multi-granularity representation enhancement ##



## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name GTpro python=3.7
$ conda activate GTpro

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of GTpro
$ git clone https://github.com/zzr624663649/GTpro.git
$ cd GTpro
```

### Dataset

You can download the pre-training data used in the paper [here](https://www.ebi.ac.uk/chembl/). All the databases for fine-tuning are saved in the folder under the benchmark name. 

### Pre-training

To train the GTpro, where the configurations and detailed explaination for each variable can be found in `pretrain_model.py`
```
$ python pretrain_model.py
```


### Fine-tuning 

To fine-tune the GTpro pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `finetune.py`
```
$ python finetune.py
```


## Acknowledgement

- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)
