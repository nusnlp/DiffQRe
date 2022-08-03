# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
# run ./finetune.sh --help to see all the possible options





## code for CANNARD
#export DATA_DIR=../data/data_with_domain_id/
#export OUT_DIR=../output/shared_result/cannard
#export MODEL_ENSEMBLE=False
#export DISTILL=False
#export round=round

#for seed in 9837  2378  456
#do
#python ../finetune_shared.py --learning_rate 1e-4 \
#    --data_dir $DATA_DIR \
#    --output_dir $OUT_DIR/${round}_$seed \
#    --overwrite_output_dir \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --num_train_epochs 10 \
#    --model_name_or_path  facebook/bart-base  \
#    --gpus 1 \
#    --do_train \
#    --do_predict \
#    --task translation \
#    --n_val -1 \
#    --val_check_interval 1.0 \

#done



## code for QReCC
export DATA_DIR=/home/yehai/diff-qre/data/data_with_domain_id
export OUT_DIR=../output/shared_result/qrecc
export MODEL_ENSEMBLE=False
export DISTILL=False
export round=round

for seed in 2378 9837 456
do
python ../finetune_shared.py --learning_rate 1e-4 \
    --data_dir $DATA_DIR \
    --output_dir $OUT_DIR/${round}_$seed \
    --overwrite_output_dir \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 8 \
    --model_name_or_path  facebook/bart-base  \
    --gpus 1 \
    --do_train \
    --do_predict \
    --task translation \
    --n_val -1 \
    --val_check_interval 1.0 \

done










