# Copyright 2024 the authors of NeuRAD and contributors.
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

"""Base data parser for Autonomous Driving datasets."""

from dataclasses import dataclass
from pathlib import Path
from nerfstudio.data.dataparsers.ad_dataparser import ADDataParser, ADDataParserConfig
from typing import List


@dataclass
class ADDepthDataParser(ADDataParser):
    """PandaSet DatasetParser"""

    config: ADDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        
        data_parser_output = super()._generate_dataparser_outputs(split)
        image_filenames = data_parser_output.image_filenames
        data_parser_output.metadata['depth_filenames'] = self._get_depth_filenames(image_filenames)
        data_parser_output.metadata["depth_unit_scale_factor"] = 1
        return data_parser_output

    def _get_depth_filenames(self, image_filenames: List[Path]) -> List[Path]:
        raise NotImplementedError
