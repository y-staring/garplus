import argparse
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# parse arguments
parser = argparse.ArgumentParser(description="Train TransE model on knowledge graph")
parser.add_argument('--in_path', type=str, default='/mnt/e/OpenKE/benchmarks/PPI/', help='Input data path')
parser.add_argument('--ckpt_path', type=str, default='/mnt/e/OpenKE/checkpoint/transe_PPI_UPDATE.ckpt', help='Checkpoint save path')
parser.add_argument('--dim', type=int, default=200, help='Dimension of entities and relations')
parser.add_argument('--margin', type=float, default=5.0, help='Margin for MarginLoss')
parser.add_argument('--nbatches', type=int, default=100, help='Number of batches for training')
args = parser.parse_args()

# 指向包含有 train2id.txt 和 valid2id.txt 的新目录
in_path = args.in_path

# dataloader for training
train_dataloader = TrainDataLoader(
    in_path = in_path, 
    nbatches = args.nbatches,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 0)

# dataloader for validation (验证集通常用在训练过程中做早停或指标监控)
valid_dataloader = TestDataLoader(in_path, "link", type_constrain = False)

# define the model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = args.dim, 
    p_norm = 1, 
    norm_flag = True)

# define the loss function
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = args.margin),
    batch_size = train_dataloader.get_batch_size()
)

# train the model
# 开启 opt_method、保存步骤等，并将 valid_dataloader 传给 trainer
trainer = Trainer(
    model = model, 
    data_loader = train_dataloader, 
    # train_times = 1000, 
    alpha = 1.0, 
    use_gpu = True,
    # opt_method = "adam",
    save_steps = 100,            # 每隔多少步保存一次
    checkpoint_dir = "/mnt/e/OpenKE/checkpoint"
)
trainer.run()
transe.save_checkpoint(args.ckpt_path)

# validate/test the model
transe.load_checkpoint(args.ckpt_path)
# Tester 实例化时不仅可以传入测试集，也可以传入验证集
tester = Tester(model = transe, data_loader = valid_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)