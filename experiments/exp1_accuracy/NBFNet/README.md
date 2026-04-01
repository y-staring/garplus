

## Installation ##

You may install the dependencies via either conda or pip. Generally, NBFNet works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Pip ###

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install ogb easydict pyyaml
```

## Reproduction ##

### 数据转换
使用prepare_data.py进行数据转换

To reproduce the results of NBFNet, use the following command. Alternatively, you
may use `--gpus null` to run NBFNet on a CPU. All the datasets will be automatically
downloaded in the code.
### 训练设置
通过修改my_dataset.yaml来配置训练参数

### 训练

```bash
python script/run.py -c config/knowledge_graph/my_dataset.yaml --gpus [0]
```


