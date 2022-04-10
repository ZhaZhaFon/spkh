install:
	pip install -r requirements.txt

config:
	cd /home/zzf/codebase/speakerhub/egs/librimix_informed && CUDA_VISIBLE_DEVICES=0 python TD-SpeakerBeam.py --exp_dir /home/zzf/experiment-speaker/td-speakerbeam_0410 --debug True
config-eval:
	cd /home/zzf/codebase/speakerhub/egs/librimix_informed && CUDA_VISIBLE_DEVICES=1 python TD-SpeakerBeam_eval.py --model_path /home/zzf/experiment-speaker/td-speakerbeam_0410/best_model.pth --result_dir /home/zzf/experiment-speaker/td-speakerbeam_0410/ --csv_dir /home/zzf/dataset/libri2mix-informed/wav8k/min/test --task sep_clean