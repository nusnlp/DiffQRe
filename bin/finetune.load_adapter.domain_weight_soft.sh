# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options









# qrecc:  code for training, with pre-training
#export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id
#export MODEL_ENSEMBLE=True
#export DISTILL=False
#export round=round
#export weight_for_domain_loss=2 #0.1 #1 #2 #0.1
#export OUT_DIR=../output/domain_weight_soft/qrecc/w_pre_train/domain_loss=$weight_for_domain_loss


#for seed in 456 9837 2378
#do
#python ../finetune.load_adapter.domain_weight_soft.py --learning_rate 1e-4 \
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
#    --initialize_private
#done




#cannard: code for training, with pre-training 
#export DATA_DIR=..//data/data_with_domain_id
#export MODEL_ENSEMBLE=True
#export DISTILL=False
#export round=round
#export weight_for_domain_loss=2 #0.1 #1 #2 #0.1
#export OUT_DIR=../output/domain_weight_soft/cannard/w_pre_train/domain_loss=$weight_for_domain_loss


#for seed in 9837 456 2378
#do
#python ../finetune.load_adapter.domain_weight_soft.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
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
#    --initialize_private
#done





#cannard: code for training, with pre-training, v2 
#export DATA_DIR=..//data/data_with_domain_id
#export MODEL_ENSEMBLE=True
#export DISTILL=False
#export round=round
#export weight_for_domain_loss=2 #0.1 #1 #2 #0.1
#export OUT_DIR=../output/domain_weight_soft/v2/cannard/w_pre_train/domain_loss=$weight_for_domain_loss.lr=1e-5/


#for seed in 9837 456 2378
#do
#python ../finetune.load_adapter.domain_weight_soft.py --learning_rate 1e-5 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
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
#    --initialize_private
#done





#cannard: code for training, without pre-training
export DATA_DIR=..//data/data_with_domain_id
export MODEL_ENSEMBLE=True
export DISTILL=False
export round=round
export weight_for_domain_loss=2 #0.1 #1 #2 #0.1
export OUT_DIR=../output/domain_weight_soft/cannard/wo_pre_train/domain_loss=$weight_for_domain_loss.batch=16


for seed in 9837 456 2378
do
python ../finetune.load_adapter.domain_weight_soft.py --learning_rate 1e-4 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/${round}_$seed \
    --overwrite_output_dir \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_train_epochs 10 \
    --model_name_or_path  facebook/bart-base \
    --adapter1   /home/yehai/diff-qre/seq2seq/output/hard_0_wo_pre_train_result/cannard/lr=1e-4/${round}_$seed/best_tfmr/pytorch_model.bin \
    --adapter2  /home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/cannard/lr=1e-4//${round}_$seed/best_tfmr/pytorch_model.bin \
    --adapter3  /home/yehai/diff-qre/seq2seq/output/hard_2_wo_pre_train_result/cannard/lr=1e-4//${round}_$seed/best_tfmr/pytorch_model.bin \
    --gpus 1 \
    --do_train \
    --do_predict \
    --task translation \
    --n_val -1 \
    --val_check_interval 1.0 \
    --initialize_private
done
