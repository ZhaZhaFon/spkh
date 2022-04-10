# reference
# GitHub @ SpeechFIT-BUT: https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/train.py
# paper @ ICASSP 2020: https://arxiv.org/abs/2001.08378

# modified and re-distributed by Zifeng Zhao @ Peking University, April 2022

import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import sys
sys.path.append('/home/zzf/codebase/speakerhub')
sys.path.append('/home/zzf/codebase/speakerhub/model')
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import singlesrc_neg_sisdr

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_dir",  required=True, type=str, help="path to save your experiment")
parser.add_argument("-c", "--conf",     required=False, default="./TD-SpeakerBeam.yml", type=str, help="path to your configure file (.yml)")
parser.add_argument("-r", "--resume",   required=False, default=False, type=str, help="path to your checkpoint(.ckpt)")
parser.add_argument("-d", "--debug",    required=False, default=False, type=bool, help="turn on debug mode or not")

def neg_sisdr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_sisdr(est_targets[:,0], targets[:,0]).mean()

def main(conf):
    
    print('>> 读取train_set...')
    from librimix_informed import LibriMixInformed
    train_set = LibriMixInformed(
        csv_dir=conf["data"]["train_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        debug=conf['main_args']['debug'],
    )
    print('>> 读取val_set...')
    val_set = LibriMixInformed(
        csv_dir=conf["data"]["valid_dir"],
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        segment_aux=conf["data"]["segment_aux"],
        debug=conf['main_args']['debug'],
    )

    num_workers = conf["training"]["num_workers"]
    if conf['main_args']['debug']:
        num_workers = 2
    print('>> 加载train_loader...')
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=num_workers,
        drop_last=True,
    )
    print('>> 加载val_loader...')
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=num_workers,
        drop_last=True,
    )
    conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    
    print('>> 定义网络结构...')
    from model.speakerbeam.td_speakerbeam import TimeDomainSpeakerBeam
    model = TimeDomainSpeakerBeam(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"],
        **conf["enroll"])
    
    print('>> 定义optimizer & scheduler...')
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=conf['training']['reduce_patience'])

    print('>> 建立 & 保存实验...')
    exp_dir = conf["main_args"]["exp_dir"]
    conf_name = conf['main_args']['conf'].split('/')[-1]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, conf_name)
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    
    print('>> 建立system...')
    loss_func = neg_sisdr_loss_wrapper
    from model.speakerbeam.system import SystemInformed
    system = SystemInformed(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )
    
    print('>> 设置callbacks...')
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1, verbose=True, save_last=True)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=conf['training']['stop_patience'], verbose=True))
    callbacks.append(LearningRateMonitor())
    
    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None
    
    resume_from_checkpoint = None
    if conf['main_args']['resume']:
        resume_from_checkpoint = os.path.join(conf['main_args']['resume'], 'last.ckpt')
        print(f'>> 加载断点 {resume_from_checkpoint}...')
    print('>> 加载Trainer...')
    max_epoch = conf["training"]["epochs"]
    if conf['main_args']['debug']:
        max_epoch = 10
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        callbacks=callbacks,
        checkpoint_callback=checkpoint,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    print('')
    print('### START TRAINING ###')
    print('')
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))

    print(f'>> trainer.fit()完成 best_model保存到{os.path.join(exp_dir, "best_model.pth")}')
    print('')
    print('### TRAINING COMPLETED ###')
    print('')
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()
    print('')
    print(f'>> 解析超参数 {args.conf}...')
    with open(args.conf) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
