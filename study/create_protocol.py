import os
import json
import hydra
from omegaconf import DictConfig
from study.protocols.protocol import Protocol


@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def create_protocol(cfg: DictConfig) -> None:
    if "protocol_name" not in cfg:
        raise ValueError("Specify the filename of the protocol to create as '+protocol_name=<FILENAME>'")
    
    # Load protocol in <PROJECT_DIR>/study/protocols/protocol_confs/{cfg.protocol_name}.json
    protocol = Protocol.from_conf(project_dir=cfg.project_dir, protocol_name=cfg.protocol_name)
    
    # Store protocol commands (<PROJECT_DIR>/study/protocols/commands/{cfg.protocol_name}.json)
    # and slurm script (<PROJECT_DIR>/study/protocols/slurm/{cfg.protocol_name}.json)
    protocol.store_commands(project_dir=cfg.project_dir)


if __name__ == "__main__":
    create_protocol()
