# accelerate config
root_path='/home/hmi/Downloads/renders/'
csv_valid_path='/mnt/hmi/thuong/Photoface_dist/geowizard_photoface_TrainValTest/dataset_0/test.csv'
output_valid_dir='/media/hmi/Transcend/geowizard_checkpoints/validation_results'
pretrained_model_name_or_path="stabilityai/stable-diffusion-2"
fined_tune_from_checkpoint='/media/hmi/Transcend1/geowizard_photoface_checkpoints_5/training_logs/checkpoint-2500/'
dataloader_num_workers=16
dataset_name='photoface'
seed=1234

accelerate launch --config_file ../node_config/1gpu.yaml \
                ../training/validate_photoface.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --fined_tune_from_checkpoint $fined_tune_from_checkpoint \
                  --dataset_path $root_path  \
                  --dataset_name $dataset_name \
                  --csv_valid_path $csv_valid_path \
                  --output_valid_dir $output_valid_dir \
                  --seed $seed \
                  --dataloader_num_workers $dataloader_num_workers