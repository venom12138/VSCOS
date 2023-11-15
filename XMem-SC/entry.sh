
if [ "$debug" -eq "1" ]; then

  
  echo debug=$debug
  export WANDB_MODE='offline'
  
else
  export WANDB_PROJECT=YOUR_WANDB_PROJECT_NAME
  export WANDB_DIR="$HOME/.exp/$WANDB_PROJECT/wandb/$exp_name"
  mkdir -p $WANDB_DIR
fi

[ -z "$entry_file" ] && entry_file=train_EPIC.py
[ -z "$num_GPUs" ] && num_GPUs=1
[ -z "$MASTER_PORT" ] && MASTER_PORT=24168
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --master_port $MASTER_PORT --nproc_per_node=$num_GPUs $entry_file \
    --en_wandb \
    $exp_args "${@}"

sleep 5