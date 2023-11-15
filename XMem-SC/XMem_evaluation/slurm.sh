export num_GPUs=1
export num_workers=30

. parse.sh

# bash run.sh --debug
[[ -z "$script" ]] && script=run.sh
if [ "$debug" -eq "0" ]; then
    srun --account="bbjv-delta-gpu" --partition=gpuA100x8 \
        -J yjw --nodes=1 --gres gpu:$num_GPUs --mem=60g --time=30:00:00 --cpus-per-task=$num_workers \
        bash $script
else
    export num_GPUs=1
    export num_workers=8
    bash run.sh --debug
fi
