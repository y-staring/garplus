import argparse
import openke
from openke.config import Trainer, Tester
from openke.module.model import RotatE
from openke.module.loss import SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# parse arguments
parser = argparse.ArgumentParser(description="Train RotatE model on knowledge graph")
parser.add_argument('--in_path', type=str, default='/mnt/e/OpenKE/benchmarks/PPI/', help='Input data path')
parser.add_argument('--ckpt_path', type=str, default='/mnt/e/OpenKE/checkpoint/PPI_UPDATE_rotate.ckpt', help='Checkpoint save path')
parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for training')
parser.add_argument('--train_times', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--dim', type=int, default=200, help='Dimension of entities and relations')
args = parser.parse_args()

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = args.in_path, 
	batch_size = args.batch_size,
	threads = 8,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader(args.in_path, "link", type_constrain = False)

# define the model
rotate = RotatE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.dim,
	margin = 6.0,
	epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = rotate, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 0.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.train_times, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
rotate.save_checkpoint(args.ckpt_path)

# test the model
rotate.load_checkpoint(args.ckpt_path)
tester = Tester(model = rotate, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)