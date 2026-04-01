
## Usage

### 安装

1. Install [PyTorch](https://pytorch.org/get-started/locally/)

2. Compile C++ files
```bash
cd openke
bash make.sh
```

### 数据格式要求

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .

  type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K .

### 数据格式转换
为了将我们的数据转换成能够直接使用的数据格式，使用convert_node_to_entity2id.py中的generate_train_valid_sets函数进行label=0样本的采样、数据集划分及数据转换。
```bash
python convert_node_to_entity2id.py
```

### 模型训练
```bash
#TransE模型
python /mnt/e/OpenKE/train_nevigate_graph.py --in_path /mnt/e/OpenKE/benchmarks/PPI/random/   --ckpt_path /mnt/e/OpenKE/checkpoint/PPI_random_transe.ckpt
#RotatE模型
python /mnt/e/OpenKE/train_rotate_nevigate_graph.py --in_path /mnt/e/OpenKE/benchmarks/PPI/random/ --train_times 1000  --ckpt_path /mnt/e/OpenKE/checkpoint/PPI_random_rotate.ckpt
```

### 模型评估
评测训练好的模型
```bash
python predict_triples.py
```