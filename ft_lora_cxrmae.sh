gpuid=5

CUDA_VISIBLE_DEVICES=$gpuid OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port `expr 12345 + $gpuid` main_finetune.py \
    --num_workers 10 \
    --batch_size 64 \
    --model vit_base_patch16_lora \
    --finetune ./pretrained_weights/CXR_MAE.pth \
    --epochs 400 \
    --blr 3e-3 \
    --warmup_epochs 5 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --nb_classes 14 \
    --dist_eval \
    --data_size 1% \
    --data_path /path/to/dataset/ \
    --script $0 \
    --optimizer adamw \
    --lora_rank 4 \
    --lora_pos attn \
    --note lora \
    # --resume /path/to/weights/checkpoint-best_auroc.pth \
    # --eval \
    # uncomment two lines above for inference
