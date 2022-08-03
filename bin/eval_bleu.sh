cat  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_2/test.target > /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target.cache

cat  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_result/test_generations.txt /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_result/test_generations.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_2_result/test_generations.txt > /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test_generations.cache



sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target.cache < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test_generations.cache



sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_result/test_generations.txt

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_result/test_generations.txt

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_2/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_2_result/test_generations.txt


rm /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target.cache

rm /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test_generations.cache


echo "shared_private_joint_training"

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target <   /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_private_result_by_joint_training/test_generations.txt



echo "hard_0_wo_pre_train_result:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target  <  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_wo_pre_train_result/test_generations.txt

echo "hard_1_wo_pre_train_result:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target  <  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_wo_pre_train_result/test_generations.txt

echo "hard_2_wo_pre_train_result:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_2/test.target  <  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_2_wo_pre_train_result/test_generations.txt




echo "hard_0_s_p:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_s_p/test_generations.txt 

echo "hard_1_s_p:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_s_p/test_generations.txt


echo "hard_2_s_p:"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_2/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_2_s_p/test_generations.txt




echo "hard_0_s_p_test"
sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_s_p_test/test_generations.txt


sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_0_load_adapter/test_generations.txt



sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_1_load_adapter/test_generations.txt

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_2/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_2_load_adapter/test_generations.txt




sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/hard_all_load_adapter/test_generations.txt


sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/gumble_train//test_generations.txt

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/gumble_train_wo_ce//test_generations.txt



sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target <    /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.txt



sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.txt


cat /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.hard_0.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.hard_1.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.hard_2.txt >  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.aaa.txt

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/no_gumble_train/test_generations.aaa.txt 




cat /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.hard_0.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.hard_1.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.hard_2.txt >  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.aaa.txt  


sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.aaa.txt

rm /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/shared_result/test_generations.aaa.txt  




sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/hard_1/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_1/test_generations.txt




sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/data_refine/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_0/test_generations.txt -w 4

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/data_refine/hard_1/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_1/test_generations.txt -w 4


sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/data_refine/hard_2/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_2/test_generations.txt -w 4



cat /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_0/test_generations.txt  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_1/test_generations.txt /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_2/test_generations.txt  >  /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_2/test_generations.aaaa.txt 

sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_2/test_generations.aaaa.txt

rm /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/output/data_refine/hard_2/test_generations.aaaa.txt


sacrebleu /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/data_refine/hard_0/test.target < /home/yehai/projects/codes/alibaba-feature/NLP_models_tasks/tasks/seq2seq/data/data_with_domain_id/data_refine/hard_0/private/test_generations.txt 
