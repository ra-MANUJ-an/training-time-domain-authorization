poetry run python main.py --experiment-name viggo_baseline \
    --model distilgpt2 \
    --tokenizer distilgpt2 \
    --train-batch-size 4 \
    --test-batch-size 4 \
    --learning-rate 1e-5 \
    --num-epochs 3 \
    --seed 42
