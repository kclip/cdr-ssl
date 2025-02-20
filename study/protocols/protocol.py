import os
import json
from itertools import product
from collections import namedtuple
from typing import List


KEY_LOGS_SUBDIR = "logs_subdir"
KEY_PROTOCOL_NAME = "+protocol_name"
KEY_IDX_RUN = "+idx_run"
PROTOCOL_CONFS_FOLDER = os.path.join("study", "protocols", "protocol_confs")
COMMANDS_FOLDER = os.path.join("study", "protocols", "commands")
SLURM_SCRIPTS_FOLDER = os.path.join("study", "protocols", "slurm")

Parameters = namedtuple(
    "Parameters",
    ["key", "values", "label"]
)


def _sanitize_param_value(value) -> str:
    return str(value).replace(" ", "")


def _sanitize_param_value_repr(value) -> str:
    return _sanitize_param_value(value).replace(".", "u")


def _slurm_template(protocol_name: str, n_commands: int, commands_filepath: str) -> str:
    return f'''#!/bin/bash

#SBATCH --job-name={protocol_name}
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --array=1-{n_commands}
#SBATCH --gres=gpu:1

# Init env
module load texlive/20220321-gcc-13.2.0-python-3.11.6
module load anaconda3/2021.05-gcc-13.2.0
source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
conda activate cdr

# Get command for this run
RUN_COMMAND=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {commands_filepath})

# Run
$RUN_COMMAND
'''


class Protocol(object):
    def __init__(
        self,
        protocol_name: str,
        logs_subdir: str,
        command: str,
        params_list: List[Parameters],
        n_runs: int = None
    ):
        self._protocol_name = protocol_name
        self._logs_subdir = logs_subdir
        self._command = command
        self._params_list = params_list
        self._n_runs = n_runs
    
    @classmethod
    def from_conf(cls, project_dir: str, protocol_name: str):
        filepath = os.path.join(project_dir, PROTOCOL_CONFS_FOLDER, f"{protocol_name}.json")
        with open(filepath, "r") as f:
            conf = json.load(f)
        params_list = [
            Parameters(key=p["key"], values=p["values"], label=p["label"])
            for p in conf["params_list"]
        ]
        return cls(
            protocol_name=conf["protocol_name"],
            logs_subdir=conf["logs_subdir"],
            command=conf["command"],
            params_list=params_list,
            n_runs=conf.get("n_runs", None)
        )
    
    def _get_commands(self, idx_run: int = None) -> List[str]:
        params_keys = [p.key for p in self._params_list]
        params_labels = [p.label for p in self._params_list]
        params_values_grid = [p.values for p in self._params_list]
        # Grid-search along all specified parameters
        commands_all_runs = []
        for params_values in product(*params_values_grid):
            # Get command params and run subdir
            run_subdir = ""
            command_params = ""
            for key, value, label in zip(params_keys, params_values, params_labels):
                if label is not None:
                    run_subdir += f"_{label}_{_sanitize_param_value_repr(value)}"
                command_params += f" {key}={_sanitize_param_value(value)}"
            # Add subdir conf
            if idx_run is not None:
                run_subdir += f"_run_{idx_run:03}"
            logs_subdir = os.path.join(self._logs_subdir, run_subdir[1:])
            command_params = f" {KEY_LOGS_SUBDIR}=\"{logs_subdir}\"" + command_params
            # Add run index
            if idx_run is not None:
                command_params = f" {KEY_IDX_RUN}={idx_run}" + command_params
            # Add protocol name to conf
            command_params = f" {KEY_PROTOCOL_NAME}={self._protocol_name}" + command_params
            # Get final command
            command_run = f"{self._command}{command_params}"
            commands_all_runs.append(command_run)
        return commands_all_runs
    
    def store_commands(self, project_dir: str):
        # Get commands for all runs (each run is a separate grid-search over the specified parameters)
        indices_runs = [None] if self._n_runs is None else range(self._n_runs)
        commands_all_runs = []
        for idx_run in indices_runs:
            commands_all_runs += self._get_commands(idx_run=idx_run)
        
        # Store each command as a separate line in a text file
        commands_filepath = os.path.join(project_dir, COMMANDS_FOLDER, f"{self._protocol_name}.sh")
        with open(commands_filepath, "w") as f:
            f.write('\n'.join(commands_all_runs))
        
        # Create a Slurm batch script template
        slurm_script = _slurm_template(
            protocol_name=self._protocol_name,
            n_commands=len(commands_all_runs),
            commands_filepath=str(commands_filepath)
        )
        slurm_script_filepath = os.path.join(project_dir, SLURM_SCRIPTS_FOLDER, f"{self._protocol_name}.sh")
        with open(slurm_script_filepath, "w") as f:
            f.write(slurm_script)
