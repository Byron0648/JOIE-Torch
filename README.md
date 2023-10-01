# JOIE-torch

Pytorch version of JOIE



训练例子

dbpedia数据集

```sh
python training_model2_scai2.py --kg1f ../data/dbpedia/db_insnet_train.txt --kg2f ../data/dbpedia/db_onto_small_mini.txt --modelname ./models  --method transe --bridge CMP-double --alignf ./dbpedia/db_InsType_mini.txt
```

yago数据集

```sh
python training_model2_scai2.py --kg1f ../data/yago/yago_insnet_train.txt --kg2f ../data/yago/yago_ontonet_train.txt --modelname ./models  --method transe --bridge CMP-double --alignf ../data/yago/yago_InsType_train.txt
```

三元组补全任务

```sh
python test_triples.py --modelname ./dbpedia --model "transe_CMP-double_dim1_300_dim2_100_a1_2.5_a2_1.0_m1_0.5_fold_3" --task "triple-completion" --testfile "../data/depedia/db_onto_small_test.txt" --resultfolder "dbpedia" --graph "onto"
```

实体分类任务

```sh
python test_triples.py --modelname ./dbpedia --model "transe_CMP-double_dim1_300_dim2_100_a1_2.5_a2_1.0_m1_0.5_fold_3" --task "entity-typing" --testfile "../data/depedia/db_InsType_test.txt" --resultfolder "dbpedia"
```

