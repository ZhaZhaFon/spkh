# reference
# GitHub @ SpeechFIT-BUT: https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/eval.py

# modified and re-distributed by Zifeng Zhao @ Peking University, April 2022

import os
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import numpy as np

from asteroid.metrics import get_metrics
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates

import sys
sys.path.append('/home/zzf/codebase/speakerhub')
sys.path.append('/home/zzf/codebase/speakerhub/model')

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir",    required=True, type=str, help="path to csv files for evaluation")
parser.add_argument("--task",       required=True, type=str, help="LibriMix subtasks, one of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`")
parser.add_argument("--model_path", required=True, type=str, help="Path to the model (either the best model or a checkpoint")
parser.add_argument("--result_dir", required=True, type=str, help="Directory where the eval results will be stored")

parser.add_argument("-c", "--conf",         required=False, type=str, default="./TD-SpeakerBeam.yml", help="path to your configure file (.yml)")
parser.add_argument("--from_checkpoint",    required=False, type=int, default=0, help="Model in model path is checkpoint, not final model. Default: 0")
parser.add_argument("--use_gpu",            required=False, type=bool, default=True, help="Whether to use the GPU for model execution")
parser.add_argument("-w", "--write_wav",    required=False, type=bool, default=False, help="Wrtite inferred wav or not")

COMPUTE_METRICS = ["si_sdr", "sdr", "stoi", "pesq"]

def main(conf):
    
    compute_metrics = COMPUTE_METRICS
    from model.speakerbeam.td_speakerbeam import TimeDomainSpeakerBeam
    if not conf["from_checkpoint"]:
        print('>> 从pretrained加载模型...')
        model = TimeDomainSpeakerBeam.from_pretrained(conf["model_path"])
    else:
        print('>> 从checkpoint加载模型...')
        model = TimeDomainSpeakerBeam(**conf["train_conf"]["filterbank"],
                                   **conf["train_conf"]["masknet"],
                                   **conf["train_conf"]["enroll"]
                                   ) 
        ckpt = torch.load(conf["model_path"], map_location = torch.device('cpu')) 
        state_dict = {} 
        for k in ckpt['state_dict']: 
            state_dict[k.split('.',1)[1]] = ckpt['state_dict'][k]
        model.load_state_dict(state_dict)
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    
    print('>> 读取test_set...')
    from librimix_informed import LibriMixInformed
    test_set = LibriMixInformed(
        csv_dir=conf["csv_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        segment_aux=None,
    )

    eval_save_dir = conf["result_dir"]
    if not os.path.exists(eval_save_dir):
        os.makedirs(eval_save_dir)
    series_list = []
    torch.no_grad().__enter__()
    print('')
    print('### START EVALUATION ###')
    print('')
    sisdr_i_sum = 0
    with tqdm(total=len(test_set)) as pbar:
        for idx in range(len(test_set)):
            mix, source, enroll = test_set[idx]
            mix, source, enroll = tensors_to_device([mix, source, enroll], device=model_device)
            est_source = model(mix.unsqueeze(0), enroll.unsqueeze(0))
            mix_np = mix.cpu().data.numpy()
            source_np = source.cpu().data.numpy()
            est_source_np = est_source.squeeze(0).cpu().data.numpy()
            utt_metrics = get_metrics(
                mix_np,
                source_np,
                est_source_np,
                sample_rate=conf["sample_rate"],
                metrics_list=COMPUTE_METRICS,
            )
            utt_metrics["mix_path"] = test_set.mixture_path
            est_source_np_normalized = normalize_estimates(est_source_np, mix_np)
            series_list.append(pd.Series(utt_metrics))

            if conf['write_wav']:
                local_save_dir = os.path.join(conf["result_dir"], "wav_est/")
                os.makedirs(local_save_dir, exist_ok=True)
                mixture_name = test_set.mixture_path.split('/')[-1].split('.')[0]
                sf.write(local_save_dir + f"{mixture_name}_"
                                        f"s{test_set.target_speaker_idx}.wav",
                        est_source_np_normalized[0], conf["sample_rate"])
                
            sisdr_i_sum += np.round(utt_metrics['si_sdr']-utt_metrics['input_si_sdr'], 2)
            
            pbar.set_postfix(sisdr_i=sisdr_i_sum/(idx+1))
            pbar.update(1)

    # Save all metrics to the experiment folder.
    print('>> 保存结果...')
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    print('>> 打印结果...')
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name + "_input"] = all_metrics_df[input_metric_name].mean()
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print('')
    print(f'>> 被测模型地址: {conf["model_path"]}')
    print("Overall metrics :")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)
        
    print('')
    print('### EVALUATION COMPLETED ###')
    print('')
    torch.cuda.empty_cache()

if __name__ == "__main__":
    
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    print('')
    print(f'>> 解析超参数 {args.conf}...')
    with open(args.conf) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)