import os
import json
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader

from settings import DEVICE
from src.data.pseudolabel import split_and_pseudolabel_dataset
from src.utils.save import SafeOpen, JSONCustomEncoder
from src.utils.utils import iterate_on_chunks
from src.train.utils import zip_cycle
from src.train.loss import compute_dr_loss
from study.utils.loaders import load_model, load_dataset, load_loss, load_optimizer, load_lr_schedule, \
    load_pretrained_model, load_bias_estimate_schedule, load_metrics
from study.utils.logs import CheckpointManager, create_tensorboard_writer
from study.utils.utils import get_batch_size


@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def train_model_dr(cfg: DictConfig) -> None:
    print(f"Training model using {100 * cfg.model_training.labeled_ratio:.1f}% of the labeled dataset and doubly-robust learning...")
    logs_subdir = os.path.join(cfg.logs_dir, cfg.logs_subdir)
    
    # Load data
    print("Loading data...")
    dataset_tr, dataset_val, dataset_te = load_dataset(cfg)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=get_batch_size(dataset_val, cfg.model_training.batch_size),
        shuffle=False,
        num_workers=cfg.study.num_workers
    )
    dataloader_te = DataLoader(
        dataset_te,
        batch_size=get_batch_size(dataset_te, cfg.model_training.batch_size),
        shuffle=False,
        num_workers=cfg.study.num_workers
    )
    
    # Load pseudolabels and split training data between labeled and pseudo-labeled data
    pretrained_model = load_pretrained_model(cfg)
    l_pseudolabeled_dataset_tr, pl_pseudolabeled_dataset_tr = split_and_pseudolabel_dataset(
        original_dataset=dataset_tr,
        pretrained_model=pretrained_model,
        labeled_ratio=cfg.model_training.labeled_ratio,
        preload_pseudolabels=cfg.study.preload_pseudolabels,
        preload_pseudolabels_batch_size=get_batch_size(dataset_tr, cfg.pretrained_model.batch_size),
        preload_pseudolabels_num_workers=cfg.study.num_workers,
        deterministic_labeled_split=cfg.model_training.deterministic_labeled_split
    )
    labeled_batch_size = cfg.model_training.batch_size if (cfg.model_training.labeled_batch_size is None) else cfg.model_training.labeled_batch_size
    l_pseudolabeled_dataloader_tr = DataLoader(
        dataset=l_pseudolabeled_dataset_tr,
        batch_size=get_batch_size(l_pseudolabeled_dataset_tr, labeled_batch_size),
        num_workers=cfg.study.num_workers,
        shuffle=True
    )
    unlabeled_batch_size = cfg.model_training.batch_size if (cfg.model_training.unlabeled_batch_size is None) else cfg.model_training.unlabeled_batch_size
    pl_pseudolabeled_dataloader_tr = DataLoader(
        dataset=pl_pseudolabeled_dataset_tr,
        batch_size=get_batch_size(pl_pseudolabeled_dataset_tr, unlabeled_batch_size),
        num_workers=cfg.study.num_workers,
        shuffle=True
    )
    
    # Model
    model = load_model(cfg).to(DEVICE)

    # Setup training
    if ("reduction" in cfg.loss.kwargs) and (cfg.loss.kwargs.reduction != "none"):
        print("WARNING: DR loss is only valid for loss reduction set to 'none'...")
    n_epochs = cfg.model_training.n_epochs
    criterion = load_loss(cfg, reduction="none")  # Force loss reduction to "sum"
    val_criterion = load_loss(cfg, is_val_loss=True)
    metrics_modules = load_metrics(cfg)
    optimizer = load_optimizer(cfg, model.parameters())
    scheduler = load_lr_schedule(cfg, optimizer)
    bias_estimate_schedule = load_bias_estimate_schedule(cfg)
    tb_writer = create_tensorboard_writer(tb_dir=cfg.tb_dir)
    checkpoint_manager = CheckpointManager(
        logs_subdir,
        checkpoint_freq=cfg.model_training.checkpoint_freq,
        log_best=cfg.model_training.checkpoint_best
    )
    
    # Store training and validation metrics
    train_info = {
        "epoch": [],
        "batch": [],
        "loss": []
    }
    validation_info = {
        "epoch": [],
        "loss": [],
        "metrics": {
            cfg_metric.type: []
            for cfg_metric in cfg.metrics
        }
    }

    # Training loop
    print("Starting training...")
    n_batches_per_epoch = max(len(l_pseudolabeled_dataloader_tr), len(pl_pseudolabeled_dataloader_tr))
    freq_tr_output = max(n_batches_per_epoch // cfg.study.n_val_displays_per_epoch, 1)
    global_step = 0
    chunk_size = cfg.model_training.batch_size if cfg.study.chunk_size is None else cfg.study.chunk_size
    for epoch in range(n_epochs):
        # Train model
        model.train()
        total_running_loss = 0.0
        total_epoch_loss_tr = 0.0
        bias_estimate_schedule_value = bias_estimate_schedule.get_value(epoch)
        tb_writer.add_scalar("schedule/bias_estimate_schedule_value", bias_estimate_schedule_value, global_step=global_step)
        for (
            batch_idx, (
                (l_inputs, l_labels, l_contexts, l_pseudolabels),
                (pl_inputs, pl_contexts, pl_pseudolabels)
            )
        ) in enumerate(zip_cycle(l_pseudolabeled_dataloader_tr, pl_pseudolabeled_dataloader_tr)):
            l_inputs, l_labels, l_contexts, l_pseudolabels = l_inputs.to(DEVICE), l_labels.to(DEVICE), l_contexts.to(DEVICE), l_pseudolabels.to(DEVICE)
            pl_inputs, pl_contexts, pl_pseudolabels = pl_inputs.to(DEVICE), pl_contexts.to(DEVICE), pl_pseudolabels.to(DEVICE)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Split forward pass into chunks and accumulate gradients
            n_labeled = len(l_inputs)
            n_pseudolabeled = len(pl_inputs)
            batch_loss = 0.0
            for (
                l_inputs_chunk, l_labels_chunk, l_contexts_chunk, l_pseudolabels_chunk,
                pl_inputs_chunk, pl_contexts_chunk, pl_pseudolabels_chunk
            ) in iterate_on_chunks(
                [
                    l_inputs, l_labels, l_contexts, l_pseudolabels,
                    pl_inputs, pl_contexts, pl_pseudolabels
                ],
                n_chunks=int(np.ceil(max(l_inputs.shape[0], pl_inputs.shape[0]) / chunk_size)),
                dim=0
            ):
                # Forward pass
                loss = compute_dr_loss(
                    model=model, criterion=criterion,
                    l_inputs=l_inputs_chunk, l_labels=l_labels_chunk, l_pseudolabels=l_pseudolabels_chunk,
                    pl_inputs=pl_inputs_chunk, pl_pseudolabels=pl_pseudolabels_chunk,
                    bias_estimate_schedule_value=bias_estimate_schedule_value,
                    n_labeled=n_labeled, n_pseudolabeled=n_pseudolabeled
                )
                
                # Accumulate gradients
                loss.backward()
                
                # Store total training loss
                batch_loss += loss.item()
                total_epoch_loss_tr += loss.item()
            total_running_loss += batch_loss
            
            # Store training info
            train_info["epoch"].append(epoch)
            train_info["batch"].append(batch_idx)
            train_info["loss"].append(round(batch_loss, 4))
            
            # GD step
            optimizer.step()
            
            # Print training stats
            if (batch_idx % freq_tr_output) == (freq_tr_output - 1):
                running_loss = total_running_loss / freq_tr_output
                tb_writer.add_scalar("loss/train", running_loss, global_step=global_step)
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss:.3f}')
                total_running_loss = 0.0
            global_step += len(l_inputs) + len(pl_inputs)

        # Step schedulers
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate validation loss/metrics
        model.eval()
        all_preds_val = []
        all_labels_val = []
        for inputs, labels, _ in dataloader_val:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)       
            with torch.no_grad():
                preds = model(inputs)
            all_preds_val.append(preds)
            all_labels_val.append(labels)
        all_preds_val = torch.cat(all_preds_val, dim=0)
        all_labels_val = torch.cat(all_labels_val, dim=0)
        val_loss = val_criterion(all_preds_val, all_labels_val)
        val_metrics = {
            cfg_metric.type: metric(all_preds_val, all_labels_val)
            for cfg_metric, metric in zip(cfg.metrics, metrics_modules)
        }
        
        # Write epoch info to Tensorboard
        tb_writer.add_scalar("validation_loss", val_loss, global_step=global_step)
        for metric_name, metric_value in val_metrics.items():
            tb_writer.add_scalar(f"validation_metric/{metric_name}", metric_value, global_step=global_step)
        
        # Store training info
        validation_info["epoch"].append(epoch)
        validation_info["loss"].append(val_loss.cpu().numpy().tolist())
        for metric_name, metric_value in val_metrics.items():
            validation_info["metrics"][metric_name].append(metric_value.cpu().numpy().tolist())
        
        # Checkpoint
        mean_train_loss = total_epoch_loss_tr / n_batches_per_epoch
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            loss_tr=mean_train_loss,
            loss_val=val_loss,
            overwrite=cfg.study.overwrite
        )

        # Display progress
        current_lr = cfg.optimizer.lr if scheduler is None else scheduler.get_last_lr()[0]
        print(
            f"========================================\n" +
            f"Epoch [{epoch+1}/{n_epochs}] \n" +
            f"LR: {current_lr:.6f} \n" +
            f"Bias estimate schedule: {bias_estimate_schedule_value:.2f} \n" +
            f"Train Loss: {mean_train_loss:.4f} \n" +
            f"Validation Loss: {val_loss:.4f} \n" +
            f"Validation Metrics: \n" +
            "\n".join([f"    - {metric_name}: {metric_value:.4f}" for metric_name, metric_value in val_metrics.items()]) + "\n"
            f"========================================\n"
        )
        
    print('...Finished Training !')
    
    # Store best-validation model
    best_checkpoint = checkpoint_manager.load_best_checkpoint()
    best_checkpoint_epoch = best_checkpoint["epoch"]
    model_filepath = os.path.join(logs_subdir, "trained_model.pth")
    with SafeOpen(model_filepath, "wb", overwrite=True) as f:
        torch.save(best_checkpoint["model_state_dict"], f)    
    
    # Load best-validation model
    model = load_model(cfg).to(DEVICE)
    model_state_dict = torch.load(model_filepath, weights_only=True)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Test best-validation model
    all_preds_te = []
    all_labels_te = []
    for batch_idx, (inputs, labels, context) in enumerate(dataloader_te):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            preds = model(inputs)
        all_preds_te.append(preds)
        all_labels_te.append(labels)
    all_preds_te = torch.cat(all_preds_te, dim=0)
    all_labels_te = torch.cat(all_labels_te, dim=0)
    test_metrics = {
        cfg_metric.type: metric(all_preds_te, all_labels_te).cpu().item()
        for cfg_metric, metric in zip(cfg.metrics, metrics_modules)
    }
    for metric_name, metric_value in test_metrics.items():
        tb_writer.add_scalar(f"test_metric/{metric_name}", metric_value)
    
    # Store run info
    run_info = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "info": {
            "best_checkpoint_epoch": best_checkpoint_epoch
        },
        "train": train_info,
        "validation": validation_info,
        "test": {
            "metrics": test_metrics
        }
    }
    if hasattr(cfg, "protocol_name"):  # Run launched from protocol
        run_info["protocol_name"] = cfg.protocol_name
    info_filepath = os.path.join(logs_subdir, "run_info.json")
    with SafeOpen(info_filepath, "w", overwrite=cfg.study.overwrite) as f:
        json.dump(run_info, f, cls=JSONCustomEncoder)


if __name__ == "__main__":
    train_model_dr()
