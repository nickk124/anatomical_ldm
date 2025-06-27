"""
custom dataset classes for medical images, modified from the original LSUN dataset,
with segmentation loading
"""
import os

import PIL.Image
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MedicalImagesBase(Dataset):
    def __init__(self,
                 data_root,
                 size=256,
                 interpolation="bicubic",
                 load_segmentations=True,
                 segmentation_root=None,
                 n_labels=None,
                 degradation=None,
                 ):
        self.data_root = data_root
        self.image_paths = [f for f in os.listdir(data_root) if f.endswith(".jpg") or f.endswith(".png")] 
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {
                              "bicubic": PIL.Image.BICUBIC,
                              }[interpolation]

        self.load_segmentations = load_segmentations
        self.segmentation_root = segmentation_root
        self.n_labels = n_labels

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        image = image.convert("L")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        if self.load_segmentations:
            seg_path = os.path.join(self.segmentation_root, example["relative_file_path_"])
            if os.path.exists(seg_path):
                seg = Image.open(seg_path)
                seg = seg.convert("L")
                seg = np.array(seg).astype(np.uint8)
                seg = Image.fromarray(seg)
                if self.size is not None:
                    seg = seg.resize((self.size, self.size), resample=PIL.Image.NEAREST)
                seg = np.array(seg).astype(np.uint8)
                
                # For anatomical LDM, we need class indices (not one-hot)
                # This allows the model to work with cross-entropy loss
                example["segmentation"] = seg.astype(np.int64)
                
                # Also provide one-hot for backward compatibility
                onehot = np.eye(self.n_labels)[seg]
                example["segmentation_onehot"] = onehot
            else:
                # Handle missing segmentation gracefully for inference
                print(f"Warning: Segmentation file not found: {seg_path}")
                example["segmentation"] = None
                example["segmentation_onehot"] = None

        return example


class BreastMRITrain(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../breast_mri_data/prior_work_1k/mri_data_labeled2D/train",
            segmentation_root="../../breast_mri_data/prior_work_1k/segmentations2D_combined/combined/train",
            n_labels=4,
            **kwargs)


class BreastMRIValidation(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../breast_mri_data/prior_work_1k/mri_data_labeled2D/val",
            segmentation_root="../../breast_mri_data/prior_work_1k/segmentations2D_combined/combined/val",
            n_labels=4,
            **kwargs)

class BreastMRITest(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../breast_mri_data/prior_work_1k/mri_data_labeled2D/test",
            segmentation_root="../../breast_mri_data/prior_work_1k/segmentations2D_combined/combined/test",
            n_labels=4,
            **kwargs)

class CTOrganTrain(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../other_data/ct_organ_large/images/train",
            segmentation_root="../../other_data/ct_organ_large/masks/all/train",
            n_labels=6,
            **kwargs)


class CTOrganValidation(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../other_data/ct_organ_large/images/val",
            segmentation_root="../../other_data/ct_organ_large/masks/all/val",
            n_labels=6,
            **kwargs)

class CTOrganTest(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../other_data/ct_organ_large/images/test",
            segmentation_root="../../other_data/ct_organ_large/masks/all/test",
            n_labels=6,
            **kwargs)


class CTOrganSmallTrain(MedicalImagesBase):
    def __init__(self, **kwargs):
        super().__init__(
            data_root="../../other_data/ct_organ/images/train_subset_1000",
            segmentation_root="../../other_data/ct_organ/masks/all/train_subset_1000",
            n_labels=6,
            **kwargs)