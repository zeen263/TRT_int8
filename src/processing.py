# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

import numpy as np
from PIL import Image


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def resize_(image, channels=3, height=608, width=608):
    # Get the image in CHW format
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    img_data = np.asarray(resized_image).astype(np.float32)

    if len(img_data.shape) == 2:
        # For images without a channel dimension, we stack
        img_data = np.stack([img_data] * 3)
        logger.debug("Received grayscale image. Reshaped to {:}".format(img_data.shape))
    else:
        img_data = img_data.transpose([2, 0, 1])  # RGB -> BGR

    assert img_data.shape[0] == channels
    return img_data

