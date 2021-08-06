#! /bin/bash
gpu=1
train_path=../wikievents/train.json #../gtt_data/train.json
dev_path=../wikievents/dev.json
test_path=../wikievents/test.json
epochs=1
batch_size=1
max_token_len=2000
workers=12
model_path=allenai/longformer-base-4096
warmup_steps=20
save_dir=checkpoints
save_filename=best-checkpoint

python main.py --train_path ${train_path} --dev_path ${dev_path} --test_path ${test_path} \
--gpu ${gpu} --epochs ${epochs} --batch_size ${batch_size} --max_token_len ${max_token_len} \
--workers ${workers} --model_path ${model_path} --warmup_steps ${warmup_steps}