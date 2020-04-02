import numpy as np

import torch.utils

from detectron2.data import get_detection_dataset_dicts, DatasetFromList
from detectron2.data import MapDataset, samplers
from detectron2.utils.env import seed_all_rng
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from kernelphysiology.dl.pytorch.datasets.dataset_mapper import DatasetMapper


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def setup(opts, cfg_file=None):
    """
    Create configs and perform basic setups.
    """
    from kernelphysiology.dl.pytorch.configs.defaults import _C
    cfg = _C.clone()
    if cfg_file is not None:
        cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def get_coco_test(batch_size, opts, cfg_file):
    cfg = setup(opts, cfg_file)
    dataset_name = 'coco_2017_val_panoptic_separated'
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=None,
    )

    dataset = DatasetFromList(dataset_dicts)
    mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def get_coco_train(batch_size, opts, cfg_file):
    cfg = setup(opts, cfg_file)
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    dataset = DatasetFromList(dataset_dicts, copy=False)

    mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.TrainingSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    # drop_last so the batch always have the same size
    train_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )

    return train_loader


def get_panoptic_network(opts, cfg_file, net_path):
    cfg = setup(opts, cfg_file)
    net = build_model(cfg)
    checkpointer = DetectionCheckpointer(net)
    checkpointer.load(net_path)
    net.train()
    return net
