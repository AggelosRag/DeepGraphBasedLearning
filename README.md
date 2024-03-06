# DGL2024 Brain Graph Super-Resolution Challenge

## Contributors

- Kevin Mancini
- Angelos Ragkousis
- Adam Tlemsani
- Kyveli Tsioli
- Mikolaj Deja

## Problem Description

- TODO Add the problem description as a summary from the report 

## Name of your model - Methodology

- TODO Add the name of the model and the methodology from the report

- TODO add Figure of your model.

## Used External Libraries

- PyTorch
- PyTorch Geometric

To install the required libraries, run the following command:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the code, one should execute the 'main.py' file. There is also a hyperparameter search file, 'hyperparam.py', 
that can be used to find the best hyperparameters for the model. It uses `wandb` to log the results of the hyperparameter.
Note that the `wandb` api key should be set at the top of the file. 

## Results

- TODO add bar plots


## References

- The model that served as a starting point for our implementation is the AGSRNet from
M. Isallari and I. Rekik. Brain graph super-resolution using adversarial graph neural network with application to 
functional brain connectivity. _Medical Image Analysis_, 71:102084, 2021. The code was adapted from the original 
implementation available at https://github.com/basiralab/AGSRNet.
- The convolutional layer used in our model is the SSGConv from [pytorch_geometric.nn.conv.SSGConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SSGConv.html?highlight=ssg#torch_geometric.nn.conv.SSGConv)
and the U-Net is from [torch_geometric.nn.models.GraphUNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphUNet.html#torch_geometric.nn.models.GraphUNet).