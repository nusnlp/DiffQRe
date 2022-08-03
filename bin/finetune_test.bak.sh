# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options
export DATA_DIR=../data/data_with_domain_id/
export OUT_DIR=../output/gumble_train/
#export CUDA_VISIBLE_DEVICES=0
export MODEL_ENSEMBLE=False
#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 20 \
#    --model_name_or_path /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/for_debug_result_8/best_tfmr \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val 1000 \
#    --val_check_interval 0.1 


#python ../finetune.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/for_debug_result_4/best_tfmr \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val 1000 \
#    --val_check_interval 0.1


#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR \
#    --overwrite_output_dir \
#   --train_batch_size 16 \
##    --eval_batch_size 16 \
#    --num_train_epochs 20 \
#    --model_name_or_path /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/for_debug_result_6/best_tfmr \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val 1000 \
#    --val_check_interval 0.1




#shared
#export DATA_DIR=../data/data_with_domain_id/
#export OUT_DIR=../output/shared_result/another-test/round_2378
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
#export DATA_DIR=../data/data_with_domain_id/
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/w_pre_train/domain_loss=2/round_2378/
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

# code for training, larger batch 
#export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id
#export MODEL_ENSEMBLE=True
#export DISTILL=False
#export round=round
#export weight_for_domain_loss=2 #0.1
#export OUT_DIR=../output/domain_weight_soft/qrecc/w_pre_train/domain_loss=$weight_for_domain_loss


#for seed in 2378 #9837 456
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
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \
#    --initialize_private
#done















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
#export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_2_w_pre_train_result/qrecc/lr=1e-5/round_456 #9837 #2378 #456 #9837 #2378 #456   #2378 #9837
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


## for cannard
export DATA_DIR=../data/data_with_domain_id/
seed=456
hard=2
export OUT_DIR=/home/yehai/diff-qre/seq2seq/output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_${seed}
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
    --val_check_interval 1.0 \





#export DATA_DIR=../data/data_with_domain_id/hard_2/
#export OUT_DIR=../output/shared_result/
#export CUDA_VISIBLE_DEVICES=0
#export MODEL_ENSEMBLE=False
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



#export DATA_DIR=../data/data_with_domain_id/
#export OUT_DIR=../output/
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#export round=round


#for p in 0 1 2
#do
#for seed in 2378 9837 456
#do
#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/hard_${p}_wo_pre_train_result/${round}_$seed/ \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path   facebook/bart-base \
#    --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0
#done
#done




#export DATA_DIR=../data/data_with_domain_id/
#export OUT_DIR=../output/
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#export round=round


#for p in 0 1 2
#do
#for seed in 2378 9837 456
#do
#python ../finetune_test.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/shared_result/${round}_$seed/ \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#   --model_name_or_path   facebook/bart-base \
#   --gpus 1 \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0
#done
#done
