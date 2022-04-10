
tdspeakerbeam-train:
	rm -rf /home/zzf/experiment-speaker/td-speakerbeam_0410
	CUDA_VISIBLE_DEVICES=1 python TD-SpeakerBeam.py --exp_dir /home/zzf/experiment-speaker/td-speakerbeam_0410
tdspeakerbeam-eval:
	CUDA_VISIBLE_DEVICES=1 python TD-SpeakerBeam_eval.py --model_path /home/zzf/experiment-speaker/td-speakerbeam_0410/best_model.pth --result_dir /home/zzf/experiment-speaker/td-speakerbeam_0410/ --csv_dir /home/zzf/dataset/libri2mix-informed/wav8k/min/test --task sep_clean