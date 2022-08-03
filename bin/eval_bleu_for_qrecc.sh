


#shared
seed=456 #2378
echo shard seed=$seed
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/test.target -i /home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/round_$seed/test_generations.txt --smooth-method none 

echo shared seed=$seed hard=0
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/round_$seed/test_generations.hard_0.txt

echo shared seed=$seed hard=1
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_1/test.target -i  /home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/round_$seed/test_generations.hard_1.txt

echo shared seed=$seed hard=2
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_2/test.target -i  /home/yehai/diff-qre/seq2seq/output/shared_result/qrecc/round_$seed/test_generations.hard_2.txt





#adapter hard 0
seed=456 #9837 #2378
domain=1
echo model=adapter-hard-$domain domain=hard-$domain seed=$seed  

sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_$domain/test.target -i /home/yehai/diff-qre/seq2seq/output/hard_${domain}_wo_pre_train_result/qrecc/round_$seed/test_generations.txt


/home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-5/round_2378
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-5/round_2378/test_generations.txt


sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-6/round_2378/test_generations.txt


/home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/test.target -i /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837/test_generations.txt





/home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837

echo shard seed=$seed
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/test.target -i /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837/test_generations.txt

echo shared seed=$seed hard=0
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_0/test.target -i /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837/test_generations.hard_0.txt

echo shared seed=$seed hard=1
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_1/test.target -i  /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837/test_generations.hard_1.txt

echo shared seed=$seed hard=2
sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/hard_2/test.target -i  /home/yehai/diff-qre/seq2seq/output/domain_weight_soft/qrecc/round_9837/test_generations.hard_2.txt


/home/yehai/diff-qre/seq2seq/output/hard_${domain}_wo_pre_train_result/qrecc/round_$seed/test_generations.txt

cat /home/yehai/diff-qre/seq2seq/output/hard_0_wo_pre_train_result/qrecc/round_9837/test_generations.txt  /home/yehai/diff-qre/seq2seq/output/hard_1_wo_pre_train_result/qrecc/round_9837/test_generations.txt /home/yehai/diff-qre/seq2seq/output/hard_2_wo_pre_train_result/qrecc/round_9837/test_generations.txt > cache 

sacrebleu /home/yehai/diff-qre/data/data_with_domain_id/test.target -i cache --smooth-method none
rm cache






/home/yehai/diff-qre/seq2seq/output/hard_0_w_pre_train_result/qrecc/lr=1e-5/round_2378
