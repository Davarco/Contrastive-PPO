# Baseline Experiments
python ppo/train.py --n_levels 50 --curl_steps 0 --tensorboard --run_name baseline_50
python ppo/train.py --n_levels 100 --curl_steps 0 --tensorboard --run_name baseline_100
python ppo/train.py --n_levels 250 --curl_steps 0 --tensorboard --run_name baseline_250
python ppo/train.py --n_levels 500 --curl_steps 0 --tensorboard --run_name baseline_500

# Contrastive Pretraining - Rotations
python ppo/train.py --n_levels 50 --curl_rotate --tensorboard --run_name rot_50
python ppo/train.py --n_levels 100 --curl_rotate --tensorboard --run_name rot_100
python ppo/train.py --n_levels 250 --curl_rotate --tensorboard --run_name rot_250
python ppo/train.py --n_levels 500 --curl_rotate --tensorboard --run_name rot_500

# Contrastive Pretraining - Cropping
python ppo/train.py --n_levels 50 --curl_crop --tensorboard --run_name crop_50
python ppo/train.py --n_levels 100 --curl_crop --tensorboard --run_name crop_100
python ppo/train.py --n_levels 250 --curl_crop --tensorboard --run_name crop_250
python ppo/train.py --n_levels 500 --curl_crop --tensorboard --run_name crop_500

# Contrastive Pretraining - Cropping + Color Distortion
python ppo/train.py --n_levels 50 --curl_crop --curl_color --tensorboard --run_name crop_color_50
python ppo/train.py --n_levels 100 --curl_crop --curl_color --tensorboard --run_name crop_color_100
python ppo/train.py --n_levels 250 --curl_crop --curl_color --tensorboard --run_name crop_color_250
python ppo/train.py --n_levels 500 --curl_crop --curl_color --tensorboard --run_name crop_color_500

# Evaluate experiments in order
python ppo/eval.py --model_path models/baseline_50.pt
python ppo/eval.py --model_path models/baseline_100.pt
python ppo/eval.py --model_path models/baseline_250.pt
python ppo/eval.py --model_path models/baseline_500.pt
python ppo/eval.py --model_path models/rot_50.pt
python ppo/eval.py --model_path models/rot_100.pt
python ppo/eval.py --model_path models/rot_250.pt
python ppo/eval.py --model_path models/rot_500.pt
python ppo/eval.py --model_path models/crop_50.pt
python ppo/eval.py --model_path models/crop_100.pt
python ppo/eval.py --model_path models/crop_250.pt
python ppo/eval.py --model_path models/crop_500.pt
python ppo/eval.py --model_path models/crop_color_50.pt
python ppo/eval.py --model_path models/crop_color_100.pt
python ppo/eval.py --model_path models/crop_color_250.pt
python ppo/eval.py --model_path models/crop_color_500.pt

