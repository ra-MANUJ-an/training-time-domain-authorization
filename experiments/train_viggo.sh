poetry run python main.py --experiment_name viggo_baseline \
    --model_name distilgpt2 \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --seed 42