# Context-Aware Doubly-Robust Semi-Supervised Learning


## Project structure

- [config/](config): [hydra](https://hydra.cc/docs/intro/) hyperparameter configuration files. All hyperparameters can be directly modified in the CLI using [hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [datasets/](datasets): dataset data files
- [src/](src): main codebase; implements the components necessary to semi-supervised training, such as :
  - [data/](src%2Fdata): data loaders
  - [models/](src%2Fmodels): trainable model classes
  - [pretrained/](src%2Fpretrained): pre-trained teacher models
  - [train/](src%2Ftrain): training components:
    - [bias_estimate_schedule.py](src%2Ftrain%2Fbias_estimate_schedule.py): curriculum learning schedules; specifies how teacher bias estimates are introduced in semi-supervised losses
    - [criterion.py](src%2Ftrain%2Fcriterion.py): loss functions and metrics
    - [loss.py](src%2Ftrain%2Floss.py): Doubly-Robust (DR) and Context-aware DR (CDR) semi-supervised training losses
    - [tuning.py](src%2Ftrain%2Ftuning.py): tuned DR (TDR) and CDR optimal tuning parameters computation
- [study/](study): codebase containing the experiments presented in the paper
  - [protocols/](study%2Fprotocols): grid-search protocols:
    - [commands/](study%2Fprotocols%2Fcommands): list of commands per protocol
    - [protocol_confs/](study%2Fprotocols%2Fprotocol_confs): user-specified protocol configurations
    - [slurm/](study%2Fprotocols%2Fslurm): execute protocol commands via [Slurm](https://slurm.schedmd.com/documentation.html) batch scripts (requires Slurm install)
  - [train_model_cdr.py](study%2Ftrain_model_cdr.py): TDR/CDR training
  - [train_model_dr.py](study%2Ftrain_model_dr.py): DR training
  - [train_model_supervised_erm.py](study%2Ftrain_model_supervised_erm.py): Supervised Empirical Risk Minimization (ERM) training
  - [train_model_pseudo_erm.py](study%2Ftrain_model_pseudo_erm.py): Pseudo-ERM (P-ERM) training
  - [plots_toy_example.py](study%2Fplots_toy_example.py): create `toy-example` experiment plots (see [Reproduce Experiments](#reproduce-experiments)) 
  - [plots_beamforming.py](study%2Fplots_beamforming.py): create `beamforming` experiment plots (see [Reproduce Experiments](#reproduce-experiments))
  - [create_protocol.py](study%2Fcreate_protocol.py): create custom grid-search protocols (see (see [Extend experiments](#extend-experiments))
- [logs/](logs): experimental data is stored in this folder by default. Can be changed by setting the keyword argument `logs_dir=<DIR>` to a custom directory `<DIR>` when executing experiments
- [scripts/generate_beamforming_dataset.py](scripts%2Fgenerate_beamforming_dataset.py): script used to generate the beamforming dataset from the ray-tracing data in the "[Environment Aware Communications](https://github.com/xuxiaoli-seu/Environment_Aware_Communications)" repository. The [ray-tracing data `raw.zip` file](datasets/beamforming/raw.zip) must be unzipped before executing this script.



## Reproduce Experiments

### Setup Python Environment

- Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Install conda environment by running the command:\
  ```conda env create -f ./environment.yaml```
- Source the environment by running:\
  ```conda activate cdr```

### Toy Example

- Execute `toy-example` protocols directly as:
```bash
bash ./study/protocols/commands/toy_example_supervised_erm.sh
bash ./study/protocols/commands/toy_example_pseudo_erm.sh
bash ./study/protocols/commands/toy_example_dr.sh
bash ./study/protocols/commands/toy_example_cdr.sh
```
- **OR** send protocols commands to Slurm cluster via:
```bash
sbatch ./study/protocols/slurm/toy_example_supervised.sh
sbatch ./study/protocols/slurm/toy_example_pseudo_erm.sh
sbatch ./study/protocols/slurm/toy_example_dr.sh
sbatch ./study/protocols/slurm/toy_example_cdr.sh
```

- Generate `toy-example` plots in `./logs/figures` as:
```bash
python ./study/plots_toy_example.py +experiment=toy_example logs_subdir=toy_example
```

### Beamforming

- Execute `beamforming` protocols directly as:
```bash
bash ./study/protocols/commands/beamforming_supervised_erm.sh
bash ./study/protocols/commands/beamforming_pseudo_erm.sh
bash ./study/protocols/commands/beamforming_dr.sh
bash ./study/protocols/commands/beamforming_tdr.sh
bash ./study/protocols/commands/beamforming_cdr.sh
```
- **OR** send protocols commands to Slurm cluster via:
```bash
sbatch ./study/protocols/slurm/toy_example_supervised.sh
sbatch ./study/protocols/slurm/toy_example_pseudo_erm.sh
sbatch ./study/protocols/slurm/toy_example_dr.sh
sbatch ./study/protocols/slurm/toy_example_tdr.sh
sbatch ./study/protocols/slurm/toy_example_cdr.sh
```

- Generate `beamforming` plots in `./logs/figures` as:
```bash
python ./study/plots_beamforming.py +experiment=beamforming logs_subdir=beamforming
```
