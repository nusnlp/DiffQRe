# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options


## for cannard with pre-training
#export DATA_DIR=../data/data_with_domain_id/
#seed=456
#hard=2
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_${seed}
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path   facebook/bart-base \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \


## for cannard without pre-training
export DATA_DIR=../data/data_with_domain_id/
seed=2378
hard=0
for hard in  1  ; do
for seed in  456  ; do
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_${hard}_wo_pre_train_result/cannard/lr=1e-4/round_${seed}
export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_${hard}_wo_pre_train_result/cannard/lr=1e-4/rerun/round_${seed}
export MODEL_ENSEMBLE=False
export DISTILL=False
python ../finetune_test.py --learning_rate 1e-4 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR \
    --overwrite_output_dir \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --num_train_epochs 10 \
    --model_name_or_path   facebook/bart-base \
    --gpus 1 \
    --do_predict \
    --task translation \
    --n_val -1 \
    --val_check_interval 1.0 
done
done















## for qrecc
#export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id/
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/round_9837
#export CUDA_VISIBLE_DEVICES=0
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path   facebook/bart-base \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \




## for qrecc
#export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id/

#for hard in 0 1 2 ;
#do
#for seed in 2378 9837 456 ;
#do
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_${hard}_w_pre_train_result/qrecc/lr=1e-5/round_${seed}
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#python ../finetune_test.py --learning_rate 1e-4 \
#                        --data_dir $DATA_DIR \
#                        --output_dir $OUT_DIR \
#                        --overwrite_output_dir \
#                        --train_batch_size 16 \
#                        --eval_batch_size 16 \
#                        --num_train_epochs 10 \
#                        --model_name_or_path   facebook/bart-base \
#                        --gpus 1 \
#                        --do_predict \
#                        --task translation \
#                        --n_val -1 \
#                        --val_check_interval 1.0 
#done
#done
