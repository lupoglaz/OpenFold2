import os
import sys
import argparse
import pickle
import torch
from pathlib import Path
from typing import Any, Dict

from alphafold.Data.dataset import GeneralFileData, get_stream
from alphafold.Data.pipeline import DataPipeline
from alphafold.Model.features import AlphaFoldFeatures
from alphafold.Model.alphafold import AlphaFold
from alphafold.Common import protein
from custom_config import model_config

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment


from Utils.loggers import PerformanceLoggingCallback

class ExponentialMovingAverage:
	def __init__(self, model:torch.nn.Module, decay:float) -> None:
		self.params = {}
		for k, v in model.state_dict().items():
			self.params[k] = v.clone().detach()
			self.device = v.device
		self.decay = decay
	
	def to(self, device:torch.device):
		for k, v in self.params.items():
			self.params[k] = v.to(device=device)
		self.device = device
	
	def update(self, model:torch.nn.Module):
		update_params = model.state_dict()
		with torch.no_grad():
			for param_name, stored in self.params.items():
				diff = stored - update_params[param_name]
				stored -= diff*(1.0 - self.decay)
				

	def load_state_dict(self, state_dict):
		self.params = state_dict["params"]
		self.decay = state_dict["decay"]

	def state_dict(self):
		return {"params": self.params, "decay": self.decay}
		

class AlphaFoldModule(pl.LightningModule):
	def __init__(self, config):
		super().__init__()
		self.af2features = AlphaFoldFeatures(config=config, device=None, is_training=True)
		self.af2 = AlphaFold(config=config.model, target_dim=22, msa_dim=49, extra_msa_dim=25, compute_loss=True)
		self.iter = 0
		self.ema = ExponentialMovingAverage(self.af2, 0.999)
		
	def logging(self, ret):
		for head_name in ret.keys():
			if 'loss' in ret[head_name].keys():
				loss_mean = torch.mean(self.all_gather(ret[head_name]['loss']))
				if self.trainer.is_global_zero:
					self.logger.experiment.add_scalar(f"Heads/{head_name}_loss", loss_mean.item(), self.iter)
		
		for key in ["fape", "sidechain_fape", "chi_loss", "angle_norm_loss"]:
			metric = ret['structure_module'][key]
			metric_mean = torch.mean(self.all_gather(metric))
			if self.trainer.is_global_zero:
				self.logger.experiment.add_scalar(f"Structure/Losses/{key}", metric_mean.item(), self.iter)
		
		for metric_name in ret['structure_module']['metrics'].keys():
			metric = ret['structure_module']['metrics'][metric_name]
			metric_mean = torch.mean(self.all_gather(metric))
			if self.trainer.is_global_zero:
				self.logger.experiment.add_scalar(f"Structure/Metrics/{metric_name}", metric_mean.item(), self.iter)

		
	def forward(self, feature_dict, pdb_path:Path=None):
		batch = self.af2features(feature_dict, random_seed=42)
		ret, total_loss = self.af2(batch, is_training=False)
		
		if not(pdb_path is None):
			protein_pdb = protein.from_prediction(features=batch, result=ret)
			with open(pdb_path, 'w') as f:
				f.write(protein.to_pdb(protein_pdb))

		return ret, total_loss

	def training_step(self, feature_dict, batch_idx):
		if self.af2features.device is None:
			self.af2features.device = self.device
		if self.af2features.dtype != self.dtype:
			self.af2features.dtype = self.dtype
		if self.ema.device != self.device:
			self.ema.to(self.device)
		
		# print('Trainer dtype:', self.dtype)
		# print('Feature dict atom_pos dtype:', feature_dict['all_atom_positions'].dtype)
		
		num_recycle = torch.randint(low=0, high=self.af2.config.num_recycle+1, size=(1,))
		num_recycle = self.trainer.accelerator.broadcast(num_recycle, src=0)
		
		batch = self.af2features(feature_dict, random_seed=42)
		batch = self.af2features.convert(batch)
		ret, total_loss = self.af2(batch, is_training=True, iter_num_recycling=num_recycle)
		self.logging(ret)
		self.iter += 1
		
		if(torch.isnan(total_loss) or torch.isinf(total_loss)):
			total_loss = None

		return total_loss

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.af2.parameters(), lr=1e-3, eps=1e-8)
		lin_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=0.99, total_iters=1000)
		# con_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=(50000-1000))
		# mul_scheduler = ConstantLR(optimizer, factor=0.95, total_iters=25000)
		# scheduler = SequentialLR(optimizer, [lin_scheduler, con_scheduler, mul_scheduler], milestones=[1000, 50000])
		return 	{"optimizer":optimizer, "lr_scheduler":{
										"scheduler": lin_scheduler, "interval": "step"
										}
				}
	
	def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
		self.ema.update(self.af2)
		return super().on_before_zero_grad(optimizer)
	
	def on_after_backward(self) -> None:
		return super().on_after_backward()

	def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
		checkpoint["ema"] = self.ema.state_dict()
		return super().on_save_checkpoint(checkpoint)

class DataModule(pl.LightningDataModule):
	def __init__(self, train_dataset_dir:Path, batch_size=1) -> None:
		super(DataModule, self).__init__()
		self.batch_size = batch_size
		self.data_train = GeneralFileData(train_dataset_dir, allowed_suffixes=['.pkl'])
		    
	def train_dataloader(self):
		def load_pkl(batch):
			file_path_list, = batch
			assert len(file_path_list) == 1
			with open(file_path_list[0], 'rb') as f:
				return pickle.load(f)

		return get_stream(self.data_train, batch_size=self.batch_size, process_fn=load_pkl)
	
	def test_dataloader(self):
		def load_pkl(batch):
			file_path_list, = batch
			assert len(file_path_list) == 1
			with open(file_path_list[0], 'rb') as f:
				return pickle.load(f)

		return get_stream(self.data_train, batch_size=self.batch_size, process_fn=load_pkl)

class CustomDDPPlugin(DDPPlugin):
    def configure_ddp(self):
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()
        self._model._set_static_graph()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	# parser.add_argument('-dataset_dir', default='/media/HDD/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-dataset_dir', default='/media/lupoglaz/AlphaFold2Dataset/Features', type=str)
	parser.add_argument('-log_dir', default='LogTrain', type=str)
	parser.add_argument('-model_name', default='model_tiny', type=str)
	# parser.add_argument('-model_name', default='model_small', type=str)
	# parser.add_argument('-model_name', default='model_big', type=str)
	parser.add_argument('-num_gpus', default=1, type=int) #per node
	parser.add_argument('-num_nodes', default=1, type=int)
	parser.add_argument('-num_accum', default=1, type=int)
	parser.add_argument('-max_iter', default=75000, type=int)
	parser.add_argument('-precision', default=16, type=int)
	parser.add_argument('-progress_bar', default=1, type=int)
	parser.add_argument('-deepspeed_config_path', default='deepspeed_config.json', type=str)

	args = parser.parse_args()
	args.dataset_dir = Path(args.dataset_dir)
	
	logger = TensorBoardLogger(args.log_dir, name=args.model_name)
	data = DataModule(args.dataset_dir, batch_size=1)#args.num_gpus*args.num_nodes)
	config = model_config(args.model_name)
	model = AlphaFoldModule(config)
	
	if "SLURM_JOB_ID" in os.environ:
		cluster_environment = SLURMEnvironment()
	else:
		cluster_environment = None
	
	trainer = pl.Trainer(	accelerator="gpu",
							gpus=args.num_gpus,
							logger=logger,
							max_steps=args.max_iter,
							num_nodes=args.num_nodes, 
							# strategy=CustomDDPPlugin(find_unused_parameters=False),
							strategy=DeepSpeedPlugin(config=args.deepspeed_config_path),
							accumulate_grad_batches=args.num_accum,
							gradient_clip_val=0.1,
							gradient_clip_algorithm = 'norm',
							precision=args.precision, 
							amp_backend="native",
							# amp_backend="apex",
							# amp_level='O3',
							enable_progress_bar=bool(args.progress_bar),
							# callbacks = [
							# 	PerformanceLoggingCallback(Path('perf.json'), args.num_gpus*args.num_nodes)
							# ],
							#resume_from_checkpoint = Path(args.log_dir)/Path(args.model_name)/Path('version_0')/Path("checkpoints/epoch=4-step=4684.ckpt")
 						)
	trainer.fit(model, data)
	trainer.save_checkpoint(Path(trainer.logger.log_dir)/Path("checkpoints/final.ckpt"), weights_only=True)

	# ckpt = torch.load(Path("LogTrain/tiny_config_wosv/version_0/checkpoints/final.ckpt"))
	# model.load_state_dict(ckpt["state_dict"])
	# model.to(device='cuda:0')
	# model.eval()
	# data_stream = data.test_dataloader()
	# for feature_dict in data_stream:
	# 	with torch.no_grad():
	# 		prediction_result, loss = model(feature_dict, Path('test.pdb'))
	# 		print(loss)



