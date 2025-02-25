# accelerate config
root_path='/home/hmi/Downloads/renders/'
csv_path = '/home/hmi/Downloads/csv_split/train.csv'
output_dir='/home/hmi/Downloads/geowizards'

pretrained_model_name_or_path='lemonaddie/geowizard'
train_batch_size=1
gradient_accumulation_steps=8
num_train_epochs=100
checkpointing_steps=1000
learning_rate=3e-5
lr_warmup_steps=0
dataloader_num_workers=8
tracker_project_name='pretrain_tracker'
seed=1234

accelerate launch --config_file ../node_config/1gpu.yaml \
                ../training/train_depth_normal_v2.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_path $root_path  \
                  --csv_path $csv_path \
                  --output_dir $output_dir \
                  --checkpointing_steps $checkpointing_steps \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --gradient_checkpointing \
                  --seed $seed \
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention \
                  --use_ema