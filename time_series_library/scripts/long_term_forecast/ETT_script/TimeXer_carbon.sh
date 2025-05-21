export CUDA_VISIBLE_DEVICES=0

model_name="TimeXer"

# region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]
# region="BPAT"
region=$1  # Get region name from command-line argument

# region=CISO  # Get region name from command-line argument

seq_len=24
label_len=24
pred_len=96

python -u run_carbon.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/carbon/ \
  --data_path "${region}_lifecycle_dataset.csv" \
  --model_id "carbon_${seq_len}_${pred_len}" \
  --model "${model_name}" \
  --data Carbon \
  --features MS \
  --seq_len ${seq_len} \
  --label_len ${label_len} \
  --pred_len ${pred_len} \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --des "Exp" \
  --itr 1 \
  --inverse



# export CUDA_VISIBLE_DEVICES=0

# model_name=TimeXer

# # region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]
# region="CISO"

# seq_len=24
# label_len=24
# pred_len=96


# python -u run_carbon.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/carbon/ \
#   --data_path "${region}_lifecycle_dataset.csv" \
#   --model_id "carbon_${seq_len}_${pred_len}" \
#   --model $model_name \
#   --data Carbon \
#   --features M \
#   --seq_len ${seq_len} \
#   --label_len ${label_len} \
#   --pred_len ${pred_len} \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --d_model 256 \
#   --des 'Exp' \
#   --itr 1 \
#   --inverse



# export CUDA_VISIBLE_DEVICES=0

# model_name=iTransformer

# # region_list = ["CISO", "PJM", "ERCO", "ISNE", "SE", "DE"]
# region="DE"

# seq_len=24
# label_len=24
# pred_len=96


# python -u run_carbon.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/carbon/ \
#   --data_path "${region}_lifecycle_dataset.csv" \
#   --model_id "carbon_${seq_len}_${pred_len}" \
#   --model "${model_name}" \
#   --data Carbon \
#   --features MS \
#   --seq_len ${seq_len} \
#   --label_len ${label_len} \
#   --pred_len ${pred_len} \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --inverse

