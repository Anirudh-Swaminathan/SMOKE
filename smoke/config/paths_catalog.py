import os


class NuscDatasetCatalog():
    DATA_DIR = "datasets"
    DATASETS = {
        "nusc_train": {
            "root": "nuscenes/v1.0-trainval/",
        },
        "nusc_test": {
            "root": "nuscenes/v1.0-test/",
        },

    }

    @staticmethod
    def get(name):
        if "nusc" in name:
            data_dir = NuscDatasetCatalog.DATA_DIR
            attrs = NuscDatasetCatalog.DATASETS[name]
            args = dict(
                root = os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="NuScenesDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class DatasetCatalog():
    DATA_DIR = "datasets"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/training/",
        },
        "kitti_test": {
            "root": "kitti/testing/",
        },

    }

    @staticmethod
    def get(name):
        if "kitti" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
