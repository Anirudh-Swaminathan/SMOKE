import os
from typing import List, Dict, Any
import csv
import logging
import random
import json
from io import StringIO
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB
from nuscenes.utils.splits import create_splits_logs

TYPE_ID_CONVERSION = {
    'car': 0,
    'truck':1,
    'bus':2,
    'trailer': 3,
    'construction_vehicle':4,
    'pedestrian': 5,
    'motorcycle':6,
    'bicycle':7,
    'traffic_cone':8,
    'barrier':9
}


class NuScenesDataset(Dataset):
    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(NuScenesDataset, self).__init__()
        self.root = root
        """
        self.image_dir = os.path.join(root, "image_2")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")
        """
        self.split = cfg.DATASETS.TRAIN_SPLIT if is_train else cfg.DATASETS.TEST_SPLIT
        self.is_train = is_train
        self.transforms = transforms

        """
        if self.split == "train":
            imageset_txt = os.path.join(root, "ImageSets", "train.txt")
        elif self.split == "val":
            imageset_txt = os.path.join(root, "ImageSets", "val.txt")
        elif self.split == "trainval":
            imageset_txt = os.path.join(root, "ImageSets", "trainval.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "ImageSets", "test.txt")
        else:
            raise ValueError("Invalid split!")
        
        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        """
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN
        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        #self.logger = logging.getLogger(__name__)
        #self.logger.info("Initializing nuScenes {} set with {} files loaded".format(self.split, self.num_samples))

        # NuScenes specific stuff here
        # KITTI unwanted lines not yet deleted above
        # Select subset of the data to look at.
        self.cur_path = os.getcwd()
        self.root = "./datasets/nuscenes"
        self.dataroot = os.path.join(self.cur_path, self.root)
        if self.split == "train" or self.split == "val" or self.split == "trainval":
            nusc_version = 'v1.0-trainval'
        else:
            nusc_version = 'v1.0-test'
        self.nusc = NuScenes(version=nusc_version, dataroot=self.dataroot)

        # Get assignment of scenes to splits.
        self.split_logs = create_splits_logs(self.split, self.nusc)

        # Use only the samples from the current split.
        self.sample_tokens = self._split_to_samples(self.split_logs)
        self.image_count = len(self.sample_tokens)
        self.image_count = 100
        self.sample_tokens = self.sample_tokens[:self.image_count]

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing nuScenes {} set with {} files loaded".format(self.split, self.image_count))

    def __len__(self):
        return self.image_count

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples

    def __getitem__(self, idx):
        # set defaults here
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        # sample_token based on index
        sample_token = self.sample_tokens[idx]

        # Get sample data.
        sample = self.nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        cam_front_token = sample['data'][self.cam_name]
        lidar_token = sample['data'][self.lidar_name]

        # Retrieve sensor records.
        sd_record_cam = self.nusc.get('sample_data', cam_front_token)
        sd_record_lid = self.nusc.get('sample_data', lidar_token)
        cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        # Combine transformations and convert to KITTI format.
        # Note: cam uses same conventions in KITTI and nuScenes.
        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                        inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                        inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Projection matrix.
        p_left_kitti = np.zeros((3, 4))
        p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

        # Create KITTI style transforms.
        velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
        velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

        # Check that the rotation has the same format as in KITTI.
        assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
        assert (velo_to_cam_trans[1:3] < 0).all()

        # Retrieve the token from the lidar.
        # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
        # not the camera.
        filename_cam_full = sd_record_cam['filename']

        # set the img variable to its data here
        src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
        img = Image.open(src_im_path)

        # Create calibration matrix.
        K = p_left_kitti
        K = [float(i) for i in K]
        K = np.array(K, dtype=np.float32).reshape(3, 4)
        K = K[:3, :3]

        # populate the list of object annotations for this sample
        anns = []
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

            # Get box in LIDAR frame.
            _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                selected_anntokens=[sample_annotation_token])
            box_lidar_nusc = box_lidar_nusc[0]

            # Truncated: Set all objects to 0 which means untruncated.
            truncated = 0.0

            # Occluded: Set all objects to full visibility as this information is not available in nuScenes.
            occluded = 0

            # Convert nuScenes category to nuScenes detection challenge category.
            detection_name = category_to_detection_name(sample_annotation['category_name'])

            # Skip categories that are not part of the nuScenes detection challenge.
            if detection_name is None:
                continue

            # Convert from nuScenes to KITTI box format.
            box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

            # Project 3d box to 2d box in image, ignore box if it does not fall inside.
            bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=imsize)
            if bbox_2d is None:
                continue

            # Set dummy score so we can use this file as result.
            box_cam_kitti.score = 0

            # Convert box to output string format.
            output = KittiDB.box_to_string(name=detection_name, box=box_cam_kitti, bbox_2d=bbox_2d,
                                            truncation=truncated, occlusion=occluded)
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
            if self.is_train:
                f = StringIO(output)
                reader = csv.DictReader(f, delimiter=' ', fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    if row["type"] in self.classes:
                        anns.append({
                            "class": row["type"],
                            "label": TYPE_ID_CONVERSION[row["type"]],
                            "truncation": float(row["truncated"]),
                            "occlusion": float(row["occluded"]),
                            "alpha": float(row["alpha"]),
                            "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                            "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                            "rot_y": float(row["ry"])
                        })

        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            K[0, 2] = size[0] - K[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        if not self.is_train:
            # for inference we parametrize with original size
            target = ParamsList(image_size=size,
                                is_train=self.is_train)
            target.add_field("trans_mat", trans_mat)
            target.add_field("K", K)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        for i, a in enumerate(anns):
            a = a.copy()
            cls = a["label"]

            locs = np.array(a["locations"])
            rot_y = np.array(a["rot_y"])
            if flipped:
                locs[0] *= -1
                rot_y *= -1

            point, box2d, box3d = encode_label(
                K, rot_y, a["dimensions"], locs
            )
            point = affine_transform(point, trans_mat)
            box2d[:2] = affine_transform(box2d[:2], trans_mat)
            box2d[2:] = affine_transform(box2d[2:], trans_mat)
            box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
            box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
            h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]

            if (0 < point[0] < self.output_width) and (0 < point[1] < self.output_height):
                point_int = point.astype(np.int32)
                p_offset = point - point_int
                radius = gaussian_radius(h, w)
                radius = max(0, int(radius))
                heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                cls_ids[i] = cls
                regression[i] = box3d
                proj_points[i] = point_int
                p_offsets[i] = p_offset
                dimensions[i] = np.array(a["dimensions"])
                locations[i] = locs
                rotys[i] = rot_y
                reg_mask[i] = 1 if not affine else 0
                flip_mask[i] = 1 if not affine and flipped else 0

        target = ParamsList(image_size=img.size,
                            is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("proj_p", proj_points)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("rotys", rotys)
        target.add_field("trans_mat", trans_mat)
        target.add_field("K", K)
        target.add_field("reg_mask", reg_mask)
        target.add_field("flip_mask", flip_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx
