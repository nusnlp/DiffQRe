#seed=456
#cd ../output/shared_result/cannard/round_$seed/
#python extract.py       test_generations.txt 
#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/shared_result/cannard/round_$seed/test_generations.hard_0.txt -w 2
#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target -i /home/yehai/diff-qre/seq2seq/output/shared_result/cannard/round_$seed/test_generations.hard_1.txt -w 2
#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target -i /home/yehai/diff-qre/seq2seq/output/shared_result/cannard/round_$seed/test_generations.hard_2.txt -w 2




#seed=456
#hard=0
#cd ../output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_$seed
#python extract.py         test_generations.txt
#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target  -i /home/yehai/diff-qre/seq2seq//output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_$seed/test_generations.hard_0.txt -w 2
#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target   -i /home/yehai/diff-qre/seq2seq//output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_$seed/test_generations.hard_1.txt -w 2
#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target  -i /home/yehai/diff-qre/seq2seq//output/hard_${hard}_w_pre_train_result/cannard/lr=1e-5/round_$seed/test_generations.hard_2.txt -w 2


#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target  -i /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.5.lr=1e-5/kl_ce/round_2378/test_generations.hard_0.txt -w 2


#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target   -i /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.5.lr=1e-5/kl_ce/round_2378/test_generations.hard_1.txt -w 2

#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target  -i  /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.5.lr=1e-5/kl_ce/round_2378/test_generations.hard_2.txt -w 2





#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target  -i  /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/v2/cannard/w_pre_train/domain_loss=2.lr=1e-5/round_9837/test_generations.hard_0.txt -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target   -i  /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/v2/cannard/w_pre_train/domain_loss=2.lr=1e-5/round_9837/test_generations.hard_1.txt -w 2

#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target  -i  /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/v2/cannard/w_pre_train/domain_loss=2.lr=1e-5/round_9837/test_generations.hard_2.txt -w 2



#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target  -i /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.1.lr=1e-4/kl_ce/round_2378/test_generations.hard_0.txt  -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target   -i  /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.1.lr=1e-4/kl_ce/round_2378/test_generations.hard_1.txt  -w 2

#sacrebleu  /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target  -i  /home/yehai/diff-qre/seq2seq/output/distillation/v2/cannard/lambda=0.1.lr=1e-4/kl_ce/round_2378/test_generations.hard_2.txt  -w 2

#sacrebleu  /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target -i  /home/yehai/diff-qre/seq2seq/output/hard_0_wo_pre_train_result/qrecc/round_9837/test_generations.txt  -w 2

#sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_1/test.target   -i  /home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/qrecc/round_456/test_generations.txt  -w 2

#sacrebleu  /home/yehai/diff-qre/data/data_with_domain_id/hard_2/test.target  -i /home/yehai/diff-qre/seq2seq/output/hard_2_wo_pre_train_result/qrecc/round_2378/test_generations.txt  -w 2


#seed=456
#hard=2
#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/hard_${hard}_wo_pre_train_result/cannard/lr=1e-4/round_${seed}/test_generations.hard_0.txt -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target -i /home/yehai/diff-qre/seq2seq/output/hard_${hard}_wo_pre_train_result/cannard/lr=1e-4/round_${seed}/test_generations.hard_1.txt -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target -i  /home/yehai/diff-qre/seq2seq/output/hard_${hard}_wo_pre_train_result/cannard/lr=1e-4/round_${seed}/test_generations.hard_2.txt -w 2



seed=2378
path=/home/yehai/diff-qre/seq2seq/output/distillation/cannard/wo_pretrain/lambda=0.5.batch=16/

path=/home/yehai/diff-qre/seq2seq/output/domain_weight_soft/cannard/wo_pre_train/domain_loss=2.batch=16

path=/home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/cannard/lr=1e-4/

path=/nlpg/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_wo_pre_train_result/

path=/home/yehai/diff-qre/seq2seq-norm/output/domain_weight_soft_norm/cannard/w_pre_train/domain_loss=2/

path=/home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/cannard/lr=1e-4/rerun/

path=/home/yehai/diff-qre/seq2seq-norm/output/domain_weight_soft_norm/cannard/w_pre_train/domain_loss=2

path=/home/yehai/diff-qre/seq2seq-only-pred/output/domain_weight_soft_only_pred/qrecc/w_pre_train/domain_loss=2/

path=/home/yehai/diff-qre/seq2seq-only-pred/output/domain_weight_soft_only_pred/cannard/w_pre_train/domain_loss=2/

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_0/test.target -i $path/round_${seed}/test_generations.hard_0.txt -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_1/test.target -i $path/round_${seed}/test_generations.hard_1.txt -w 2

#sacrebleu /home/yehai/diff-qre/seq2seq/data/data_with_domain_id/hard_2/test.target -i $path/round_${seed}/test_generations.hard_2.txt -w 2


seed=9837
#path=/home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/
#path=/home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-5/
#path=/home/yehai/diff-qre/seq2seq/output/hard_1_w_pre_train_result/qrecc/lr=1e-5/
#path=/home/yehai/diff-qre/seq2seq/output/hard_2_w_pre_train_result/qrecc/lr=1e-5

#path=/home/yehai/diff-qre/seq2seq/output/distillation/qrecc/lambda=0.9/
path=/home/yehai/diff-qre/seq2seq-only-pred/output/domain_weight_soft_only_pred/qrecc/w_pre_train/domain_loss=2/

path=/home/yehai/diff-qre/seq2seq-norm/output/domain_weight_soft_norm/qrecc/w_pre_train/domain_loss=2/
#path=/home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/w_pre_train/domain_loss=2/
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target  -i   $path/round_${seed}/test_generations.hard_0.txt -w 2 
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_1/test.target  -i   $path/round_${seed}/test_generations.hard_1.txt -w 2 
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_2/test.target  -i   $path/round_${seed}/test_generations.hard_2.txt -w 2 

