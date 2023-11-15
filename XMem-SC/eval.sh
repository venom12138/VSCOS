output_path=YOUR_OUTPUT_PATH
model_path=YOUR_MODEL_PATH
python eval_EPIC.py --model $model_path \
    --output $output_path \
    --use_handmsk 0 --use_flow 0 --use_text 0 --fuser_type cbam

cd ./XMem_evaluation
python evaluation_method.py --results_path $output_path/all
cd ..