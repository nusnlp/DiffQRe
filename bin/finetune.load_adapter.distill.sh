# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options



## cannard: without pre-training
export DATA_DIR=../data/data_with_domain_id/
lambda=0.5
export OUT_DIR=../output/distillation/cannard/wo_pretrain/lambda=$lambda.batch=16/
export MODEL_ENSEMBLE=False
export DISTILL=True
export round=round
for seed in 2378 9837 456
do
python ../finetune.load_adapter.distillation.py --learning_rate 1e-4 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/${round}_$seed \
    --overwrite_output_dir \
    --train_batch_size 16 \
    --eval_batch_size 10 \
    --num_train_epochs 10 \
    --model_name_or_path  facebook/bart-base \
    --adapter1  /home/yehai/diff-qre/seq2seq/output/hard_0_wo_pre_train_result/cannard/lr=1e-4/${round}_$seed/best_tfmr/pytorch_model.bin \
    --adapter2 /home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/cannard/lr=1e-4/${round}_$seed/best_tfmr/pytorch_model.bin \
    --adapter3 /home/yehai/diff-qre/seq2seq/output/hard_2_wo_pre_train_result/cannard/lr=1e-4/${round}_$seed/best_tfmr/pytorch_model.bin \
    --gpus 1 \
    --do_train \
    --do_predict \
    --task translation \
    --n_val -1 \
    --val_check_interval 1.0 \
    --_lambda $lambda \
    --initialize_private 
done

##qrecc: with pre-training
#export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id
#lambda=0.9 #0.5
#export OUT_DIR=../output/distillation/qrecc/lambda=$lambda
#export MODEL_ENSEMBLE=False
#export DISTILL=True
#export round=round
#for seed in  456  9837   2378 
#do
#python ../finetune.load_adapter.distillation.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 32 \
#    --eval_batch_size 16 \
#    --num_train_epochs 8 \
#    --model_name_or_path  facebook/bart-base \
#    --adapter1  /home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter2 /home/yehai/diff-qre/seq2seq/output/hard_1_w_pre_train_result/qrecc/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter3 /home/yehai/diff-qre/seq2seq/output/hard_2_w_pre_train_result/qrecc/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --gpus 1 \
#    --do_train \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \
#    --_lambda $lambda \
#    --initialize_private 
#done




# cannard: with pre-training
#export DATA_DIR=../data/data_with_domain_id
#lambda=0.1 #0.9 #0.5
#export OUT_DIR=../output/distillation/cannard/lambda=$lambda/kl_ce/batch=32
#export MODEL_ENSEMBLE=False
#export DISTILL=True
#export round=round
#for seed in  2378  9837  456  
#do
#python ../finetune.load_adapter.distillation.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 32 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path  facebook/bart-base \
#    --adapter1  /home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter2 /home/yehai/diff-qre/seq2seq/output/hard_1_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter3 /home/yehai/diff-qre/seq2seq/output/hard_2_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --gpus 1 \
#    --do_train \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \
#    --_lambda $lambda \
#    --initialize_private
#done




# cannard: with pre-training v2
#export DATA_DIR=../data/data_with_domain_id
#lambda=0.9 #0.9 #0.5
##export OUT_DIR=../output/distillation/v2/cannard/lambda=$lambda.lr=1e-4/kl_ce/
#export MODEL_ENSEMBLE=False
#export DISTILL=True
#export round=round
#for seed in  2378  9837  456  
#do
#python ../finetune.load_adapter.distillation.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
##    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path   /home/yehai/diff-qre/seq2seq/output/shared_result/cannard/round_$seed/best_tfmr  \
#    --adapter1  /home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter2 /home/yehai/diff-qre/seq2seq/output/hard_1_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --adapter3 /home/yehai/diff-qre/seq2seq/output/hard_2_w_pre_train_result/cannard/lr=1e-5/${round}_$seed/best_tfmr/pytorch_model.bin \
#    --gpus 1 \
#    --do_train \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \
#    --_lambda $lambda \
#    --initialize_private
#done
