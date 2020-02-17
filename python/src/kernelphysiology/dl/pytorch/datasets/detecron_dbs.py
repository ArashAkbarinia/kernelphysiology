"""

"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import CLASS_NAMES, load_voc_instances

__all__ = ["register_all_pascal_voc_org"]


def register_all_pascal_voc_org(root):
    SPLITS = [
        ("voc_org_2007_trainval", "VOC2007", "trainval"),
        ("voc_org_2007_train", "VOC2007", "train"),
        ("voc_org_2007_val", "VOC2007", "val"),
        ("voc_org_2007_test", "VOC2007", "test"),
        ("voc_org_2012_trainval", "VOC2012", "trainval"),
        ("voc_org_2012_train", "VOC2012", "train"),
        ("voc_org_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_voc_org(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_voc_org(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )
