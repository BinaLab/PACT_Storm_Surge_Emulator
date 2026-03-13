# PACT: Peak-Aware Cross-Attention Graph Transformer for Storm Surge Forecasting

This repo is the official code for PACT, a peak-aware cross-attention cross-attention graph transformer for storm surge forecasting. It supports the following core workflow:

- **Load pre-built forcing graphs once** 
- For a certain **station**, **split by YEAR** with a **proposed ratio** in the configuration file. 
- Compute **X/Y normalization stats from TRAIN only**.
- **Normalize X on GPU**
- Train with **MSE on normalized y**
- Report metrics in **physical y units**
- Select best checkpoint by **val_rmse_phys**
- Save test predictions in **UNNORMALIZED** physical units

## Project Layout

```
Emulator/
├── README.md
├── environment_training.yml              # Conda environment for training
├── environment_dataprep.yml              # Conda environment for data preparation
├── emulator/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── distributed.py                # Helper function for single-node distributed data parallel training
│   │   ├── io_utils.py                   # Helper function for writing meta json files 
│   │   └── runtimes.py                   # Runtime helpers for deterministic experiments and CPU threading control
│   ├── data/
│   │   ├── __init__.py
│   │   ├── graph_store.py                # Graph dataset storage and split/view helpers
│   │   ├── loaders.py                    # DataLoader construction utilities
│   │   ├── normalization.py              # Feature and target normalization utilities
│   │   ├── station_metadata.py           # Station metadata parsing and encoding helpers
│   │   └── stats.py                      # Training-data statistics for normalization and loss shaping
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py                     # Inference execution helpers.
│   │   └── grouping.py                   # Inference-time sample grouping helpers.
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py              # Graph and perceiver-style architectures
│   └── train/
│       ├── __init__.py  
│       ├── engine.py                     # Training and evaluation loops
│       └── losses.py                     # Loss functions used by training
├── station_json/
│   └── Battery.json                      # station metadata from the NOAA website (lat/lon/elev/etc.)
├── configs/
│   ├── configs_train/                    # Train configuration files for PACT
│   └── configs_infer/                    # Inference configuration files for PACT
├── infer.sh                              # Standard inference bash file for non-slurm runs
├── infer_multi.sh                        # Inference bash file which support multiple inference runs
├── infer_sbatch.sh                       # Slurm version of infer.sh
├── infer_sbatch_multi.sh                 # Slurm version of infer_multi.sh
├── infer.py                              # Endpoint python file for inference
├── train.sh                              # Stadard train bash file for non-slurm runs
├── train_sbatch.sh                       # Slurm version of train.sh
├── train_sbatch_multi.sh                 # Slurm train bash file which support run multiple experiments
├── train.py                              # Endpoint python file for training
├── Inference_Checkpoints/                # Checkpoints for inferences (Used in PACT)
├── preprocessing/                        # Data preprocessing and graph generation codes
└── Data/                                 # Dataset folder
    ├── Grid4_New/
    │   ├── NCEP/
    │   │   └──graphs/
    │   ├── CMIP6_AWI/
    │   ├── CMIP6_CNRM/
    │   ├── CMIP6_EC_EARTH/
    │   ├── CMIP6_MPI/
    │   └── CMIP6_MRI/
    ├── Grid4_New_PastOnly                      
    └── Grid8_New                     
```


## Models

This repo support a baseline model without historical forcing data, which contains GraphSAGE, pooling, and linear prediction head; a baseline model with historical forcing data, which contains GraphSAGE, pooling, LSTM and linear layers; PACT(Perceiver3), our proposed peak-aware cross-attention graph transformer.

## Quickstart

Use files in the preprocessing folder to pre-processing the ADCIRC simulation files and generate the Graph with Pytorch Geometric. Then put the generated files into the Data folder.

For training in non-slurm environment, run 

```bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3.sh```.

For training in slurm environment, run 

```sbatch train_sbatch.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3.sh```.

For inference in non-slurm environment, run 

```bash infer.sh configs/configs_infer/infer_config_NCEP.sh```.

For inference in slurm environment, run 

```sbatchh infer_sbatch.sh configs/configs_infer/infer_config_NCEP.sh```.

Do not forget to change the configuration files based on the path of graph data.
