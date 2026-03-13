import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures


os.environ['WANDB_API_KEY'] = 'a50f8e095a87e9115f9ea064d8620569e88bbc77'
warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    print("[DEBUG] cfg.general.gpus =", cfg.general.gpus)
    print("[DEBUG] cfg.model.extra_features =", cfg.model.extra_features)
    print("[DEBUG] cfg.dataset.num_subgraphs =", cfg.dataset.num_subgraphs)
    print(f"-------- DEBUG CHECK --------")
    print(f"Config n_epochs: {cfg.train.n_epochs}")
    print(f"Config dataset size: {cfg.dataset.num_subgraphs}")
    print(f"-----------------------------")
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
        
    # 新增数据集
    elif dataset_config["name"] == 'epinions':
        from datasets.epinions_dataset import EpinionsDataModule, EpinionsDatasetInfos
        from analysis.visualization import NonMolecularVisualization
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
        from analysis.spectre_utils import SpectreSamplingMetrics

        # ---------------------------------------------------------
        # ★★★ 修正点 1：只传 cfg，不要传一堆 kwargs ★★★
        # 所有参数逻辑都在 EpinionsDataModule.__init__ 内部处理
        # ---------------------------------------------------------
        datamodule = EpinionsDataModule(cfg)

        # Dataset Infos (负责统计分布)
        dataset_infos = EpinionsDatasetInfos(datamodule, dataset_config)

        # 训练指标
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        # 额外特征 (Extra Features)
        # 注意：如果有显存 OOM 问题，可以先把 extra_features 设为 None
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        # extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        # 计算输入输出维度
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features
        )

        # 采样指标 (Sampling Metrics)
        # 使用 SpectreSamplingMetrics 计算 Degree/Clustering 等分布距离
        sampling_metrics = SpectreSamplingMetrics(
            datamodule, 
            compute_emd=False,
            metrics_list=['degree', 'clustering', 'motif'] # 选几个关心的指标
        )

        model_kwargs = {
            'dataset_infos': dataset_infos,
            'train_metrics': train_metrics,
            'sampling_metrics': sampling_metrics,
            'visualization_tools': visualization_tools,
            'extra_features': extra_features,
            'domain_features': domain_features
        }
        
    # 新增数据集
    elif dataset_config["name"] == 'ppi':
        # ---------------------------------------------------------
        # ★★★ PPI Dataset Configuration for GAR+ ★★★
        # ---------------------------------------------------------
        # from datasets.ppi_dataset import PPIDataModule, PPIDatasetInfos  # 修正导入名称
        from datasets.ppi_dataset_order_embedding import PPIDataModule, PPIDatasetInfos  # 修正导入名称
        from analysis.visualization import NonMolecularVisualization
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
        from analysis.spectre_utils import SpectreSamplingMetrics

        # 1. 初始化 DataModule
        # 所有路径和采样参数逻辑都在 PPIDataModule.__init__ 内部处理
        datamodule = PPIDataModule(cfg)

        #========================================
        from collections import Counter
        import torch

        def count_bitmasks(dataloader, max_batches=None):
            cnt = Counter()
            for bi, data in enumerate(dataloader):
                if hasattr(data, "edge_label_mask"):
                    masks = data.edge_label_mask.detach().cpu().tolist()
                    cnt.update(masks)
                else:
                    raise ValueError("Batch里没有 edge_label_mask；你需要在Dataset里保留它。")

                if max_batches is not None and (bi + 1) >= max_batches:
                    break

            print("\n==== Bitmask stats ====")
            print("不同 bitmask（谓词组合）数量 =", len(cnt))
            print("总边数 =", sum(cnt.values()))
            print("最常见的 10 种 bitmask =", cnt.most_common(10))
            return cnt

        train_loader = datamodule.train_dataloader()

        cnt = count_bitmasks(train_loader) 
        # ===================================================
        # 2. 初始化 Dataset Infos (负责统计节点类型分布、边类型分布等)
        dataset_infos = PPIDatasetInfos(datamodule, dataset_config)

        # 3. 训练指标 (根据模型类型选择)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        # 4. 额外特征 (Extra Features)
        # 对于 PPI 无向图，Cycles 和 Eigenvalues 是捕捉功能模块的关键
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        
        # PPI 是非分子图，不需要 Domain Specific Features (如原子价态)
        domain_features = DummyExtraFeatures()

        # 5. 计算输入输出维度 (GAR+ 编码维度 + Extra Features 维度)
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features
        )

        # 6. 采样指标 (Sampling Metrics)
        # SpectreSamplingMetrics 计算 Degree/Clustering/Orbit 等分布距离
        # compute_emd=True 会稍微慢一点但更准确，metrics_list 根据需要调整
        #validty：没有太多边也没有太少边的比例（连通性和平面性）
        sampling_metrics = SpectreSamplingMetrics(
            datamodule, 
            compute_emd=True, 
            metrics_list=['degree', 'clustering', 'orbit'] 
        )

        model_kwargs = {
            'dataset_infos': dataset_infos,
            'train_metrics': train_metrics,
            'sampling_metrics': sampling_metrics,
            'visualization_tools': visualization_tools,
            'extra_features': extra_features,
            'domain_features': domain_features
        }
        
    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    import time
    start_time = time.time()
    main()
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\n⏱️ Total runtime: {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")
    # main()
