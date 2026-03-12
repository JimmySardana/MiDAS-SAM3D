# Copyright (c) Meta Platforms, Inc. and affiliates.
import io
import os

# not ideal to put that here
os.environ["LIDRA_SKIP_INIT"] = "true"

from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import shutil
import subprocess
from PIL import Image
import numpy as np
import builtins

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils


__all__ = ["Inference"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )

def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(data):
    image = Image.open(io.BytesIO(data)).convert("RGBA")
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(data):
    mask = load_image(data)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def load_single_mask(folder_path, index=0, extension=".png"):
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path, indices_list=None, extension=".png"):
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    if not len(indices_list) > 0:  # get all all masks if not provided
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1

    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    return masks

