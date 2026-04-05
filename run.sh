export WANDB_API_KEY=""
torchrun --nproc-per-node=2 --master_addr=127.0.0.1 --master_port=29500 vla-scripts/temporal_finetune.py 