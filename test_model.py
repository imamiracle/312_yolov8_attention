# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA
from ultralytics.data.build import load_inference_source
from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    LINUX,
    MACOS,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    Retry,
    checks,
    is_dir_writeable,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

CFG = "ultralytics/cfg/models/v8/my_yolov8.yaml"
SOURCE = ASSETS / "bus.jpg"



def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(SOURCE)