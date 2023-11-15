export iterations=10000
export save_network_interval=1000
export finetune=0
export exp_name=YOUR_EXP_NAME
export exp_args="--batch_size 8 --num_frames 8 --num_workers $num_workers --log_image_interval 500 --iterations $iterations --save_network_interval $save_network_interval --finetune $finetune --load_network ./saves/XMem.pth " # --load_network ./saves/XMem.pth 
pooling_stride=2
rw_loss=0.1
use_randn_walk_loss=1

export MASTER_PORT=34798
export run_name="YOUR_RUN_NAME"
bash entry.sh --epic_root ./EPIC_train --yaml_root ./EPIC_train/EPIC100_state_positive_train.yaml \
            --freeze 0 --fuser_type cbam --steps 1000 --seed $seed \
            --use_teacher_model --teacher_warmup 100 --teacher_loss_weight 0.01 \
            --use_randn_walk_loss $use_randn_walk_loss --randn_walk_droprate 0 --randn_walk_downsample 'pooling' --randn_walk_head 1 \
            --randn_walk_loss_rate $rw_loss --randn_walk_pooling_stride $pooling_stride \
            --start_warm 25000 --end_warm 150000 "${@}"
