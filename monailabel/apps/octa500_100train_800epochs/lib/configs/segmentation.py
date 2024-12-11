# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import SegResNet
from monai.utils import optional_import

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, remove_file, strtobool

# Added: for epistemic active learning strategy
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.tasks.scoring.dice import Dice

_, has_cp = optional_import("cupy")
_, has_cucim = optional_import("cucim")

logger = logging.getLogger(__name__)


class Segmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        conf_labels = self.conf.get("labels")
        self.labels = (
            {label: idx for idx, label in enumerate(conf_labels.split(","), start=1)}
            if conf_labels
            else {
                "ILM-IPL": 1,
                "IPL-OPL": 2,
                "OPL-ISOS": 3,
                "ISOS-RPE": 4,
                "RPE-BM": 5,
                "sub-BM": 6
            }
        )

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        if not conf_labels and strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/radiology_segmentation_segresnet_multilabel.pt"
            download_file(url, self.path[0])

        # Remove pre-trained pt if user is using his/her custom labels.
        if conf_labels:
            remove_file(self.path[0])

        self.target_spacing = (0.01233552, 0.01233552, 0.00625)  # target space for image
        # Setting ROI size - This is for the image padding
        self.roi_size = (128, 128, 128)

        # Network
        self.network = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(self.labels) + 1,  # labels plus background,
            init_filters=32,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )

        # Others
        # Added: for epistemic active learning strategy
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.Segmentation(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=strtobool(self.conf.get("preload", "false")),
            config={"largest_cc": True if has_cp and has_cucim else False},
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.Segmentation(
            model_dir=output_dir,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            load_path=load_path,
            publish_path=self.path[1],
            description="Train Segmentation Model",
            labels=self.labels,
        )
        return task
    
    # Added: for epistemic active learning strategy
    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=SegResNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=len(self.labels) + 1,  # labels plus background,
                    init_filters=32,
                    blocks_down=(1, 2, 2, 4),
                    blocks_up=(1, 1, 1),
                    dropout_prob=0.2,
                ),
                transforms=lib.infers.Segmentation(None).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods