# DiffQRe

## The code is for our ACL2022 paper:  [On the Robustness of Question Rewriting Systems to Questions of Varying Hardness](https://aclanthology.org/2022.acl-long.149.pdf)


## 1. Environment
##### a. conda create --name qre  python=3.7
##### b. pip install pytorch_lightning==0.8.0 or pip install pytorch_lightning==1.0.4
##### c. pip install transformers==3.3.1



## 2. to Obtain Adapter S on BART
##### sh ./bin/finetune_shared.sh


## 3. to Train Private Models
##### sh ./bin/finetune_shared.hard_0.sh
##### sh ./bin/finetune_shared.hard_1.sh
##### sh ./bin/finetune_shared.hard_2.sh


## 4. to Train SLAF
##### sh ./bin/finetune.load_adapter.domain_weight_soft.sh


## 5. to Train SLAD
##### sh ./bin/finetune.load_adapter.distill.sh


## 6. to Do Evaluation
##### sh ./bin/eval_bleu.sh


## 7. Data Downloading
##### Cannard:  [https://sites.google.com/view/qanta/projects/canard](https://sites.google.com/view/qanta/projects/canard)
##### QReCC: [https://github.com/apple/ml-qrecc](https://github.com/apple/ml-qrecc)












