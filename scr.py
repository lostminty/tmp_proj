import os
import sys
from typing import Callable, List

import torch
from torch.utils.data.sampler import RandomSampler

import flash
from flash.core.classification import Labels
from flash.core.data.utils import download_data
from flash.core.finetuning import NoFreeze
from flash.core.utilities.imports import _KORNIA_AVAILABLE, _PYTORCHVIDEO_AVAILABLE
from flash.video import VideoClassificationData, VideoClassifier

if _PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE:
    import kornia.augmentation as K
    from pytorchvideo.transforms import ApplyTransformToKey, RandomShortSideScale, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip
else:
    print("Please, run `pip install torchvideo kornia`")
    sys.exit(1)

if __name__ == '__main__':

    # 1. Download a video clip dataset. Find more dataset at https://pytorchvideo.readthedocs.io/en/latest/data.html
    #download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip")

    # 2. [Optional] Specify transforms to be used during training.
    # Flash helps you to place your transform exactly where you want.
    # Learn more at:
    # https://lightning-flash.readthedocs.io/en/latest/general/data.html#flash.core.data.process.Preprocess
    post_tensor_transform = [UniformTemporalSubsample(8), RandomShortSideScale(min_size=256, max_size=320)]
    per_batch_transform_on_device = [K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225]))]

    train_post_tensor_transform = post_tensor_transform + [RandomCrop(244), RandomHorizontalFlip(p=0.5)]
    val_post_tensor_transform = post_tensor_transform + [CenterCrop(244)]
    train_per_batch_transform_on_device = per_batch_transform_on_device

    def make_transform(
        post_tensor_transform: List[Callable] = post_tensor_transform,
        per_batch_transform_on_device: List[Callable] = per_batch_transform_on_device
    ):
        return {
            "post_tensor_transform": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(post_tensor_transform),
                ),
            ]),
            "per_batch_transform_on_device": Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=K.VideoSequential(
                        *per_batch_transform_on_device, data_format="BCTHW", same_on_frame=False
                    )
                ),
            ]),
        }

    # 3. Load the data from directories.
    datamodule = VideoClassificationData.from_folders(
    	train_folder="/home/lost/makea/train",
    	val_folder="/home/lost/makea/val",
        predict_folder="/home/lost/makea/test",
        train_transform=make_transform(train_post_tensor_transform),
        val_transform=make_transform(val_post_tensor_transform),
        predict_transform=make_transform(val_post_tensor_transform),
        batch_size=2,
        clip_sampler="uniform",
        clip_duration=1,
        video_sampler=RandomSampler,
        decode_audio=False,
        num_workers=8
    )
    
    print(datamodule.num_classes)
    # 4. List the available models
    print(VideoClassifier.available_backbones())
    # out: ['efficient_x3d_s', 'efficient_x3d_xs', ... ,slowfast_r50', 'x3d_m', 'x3d_s', 'x3d_xs']
    print(VideoClassifier.get_backbone_details("x3d_xs"))

    # 5. Build the VideoClassifier with a PyTorchVideo backbone.
    model = VideoClassifier(
        backbone="slow_r50", num_classes=datamodule.num_classes, serializer=Labels(), pretrained=False
    )

    # 6. Finetune the model
    trainer = flash.Trainer(fast_dev_run=False,gpus=1)
    trainer.finetune(model, datamodule=datamodule)

    trainer.save_checkpoint("video_classification.pt")

    # 7. Make a prediction
    predictions = model.predict(os.path.join(os.getcwd(),"makea/test"))
    print(predictions)
    # ['marching', 'flying_kite', 'archery', 'high_jump', 'bowling']

