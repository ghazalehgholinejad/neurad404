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

"""
NeuRAD (Neural Rendering for Autonomous Driving) model.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Type, Union

import nerfacc
# import kornia.core as kornia_ops
# import kornia.losses
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.neurad_field import (NeuRADField, NeuRADFieldConfig,
                                            NeuRADProposalField,
                                            NeuRADProposalFieldConfig)
from nerfstudio.model_components.cnns import BasicBlock
from nerfstudio.model_components.losses import (L1Loss, MSELoss,
                                                VGGPerceptualLossPix2Pix,
                                                distortion_loss,
                                                zipnerf_interlevel_loss)
from nerfstudio.model_components.ray_samplers import (PowerSampler,
                                                      ProposalNetworkSampler)
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   FeatureRenderer,
                                                   NormalsRenderer)
from nerfstudio.models.ad_model import ADModel, ADModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.external import TCNN_EXISTS
from nerfstudio.utils.math import chamfer_distance
from nerfstudio.utils.printing import print_tcnn_speed_warning
from nerfstudio.viewer.server.viewer_elements import (ViewerDropdown,
                                                      ViewerSlider)

EPS = 1e-7


@dataclass
class LossSettings:
    """Configuration of all the losses."""

    vgg_mult: float = 0.05
    """Multipier for VGG perceptual loss."""
    rgb_mult: float = 5.0
    """Multipier for RGB loss."""
    depth_mult: float = 0.01
    """Multipier for lidar loss."""
    intensity_mult: float = 0.1
    """Multipier for lidar intensity loss."""
    carving_mult: float = 0.01
    """Multiplier for the lidar carving loss."""
    carving_epsilon: float = 0.1
    """Epsilon for regularization losses (both carving and eikonal), in meters."""
    quantile_threshold: float = 0.95
    """Quantile threshold for lidar and regularization losses."""
    interlevel_loss_mult: float = 0.001
    """Multiplier for the interlevel loss."""
    distortion_loss_mult: float = 0.002
    """Multiplier for the distortion loss."""
    non_return_lidar_distance: float = 150.0
    """Distance threshold for the non-return carving loss (in meters)."""
    non_return_loss_mult: float = 0.1
    """Multiplier for the non-return lidar losses (on top of existing multiplier)."""
    ray_drop_loss_mult: float = 0.01
    """Multiplier for the ray drop loss."""
    prop_lidar_loss_mult: float = 0.1
    """Multiplier for the proposal depth loss (on top of existing multiplier)."""
    shadow_mult : float = 0.1


@dataclass
class SamplingSettings:
    """Configuration of NeuRADs proposal sampling."""

    single_jitter: bool = True
    """Use the same random jitter for all samples in the same ray (in training)."""

    proposal_field_1: NeuRADProposalFieldConfig = field(default_factory=NeuRADProposalFieldConfig)
    """First proposal field configuration."""
    proposal_field_2: NeuRADProposalFieldConfig = field(default_factory=NeuRADProposalFieldConfig)
    """Second proposal field configuration."""
    num_proposal_samples: Tuple[int, ...] = (128, 64)
    """Number of proposal samples per ray."""
    num_nerf_samples: int = 32
    """Number of nerf samples per ray."""
    power_lambda: float = -1.0
    """Lambda for the distance power function."""
    power_scaling: float = 0.1
    """Scaling for the distance power function."""
    sky_distance: float = 20000.0
    """Distance to use for the sky field."""


@dataclass
class NeuRADModelConfig(ADModelConfig):
    """Overall configuration of NeuRAD."""

    _target: Type = field(default_factory=lambda: NeuRADModel)
    """Target class to instantiate."""

    loss: LossSettings = field(default_factory=LossSettings)
    """All the loss parameters."""
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    """All the sampling parameters."""
    field: NeuRADFieldConfig = field(default_factory=NeuRADFieldConfig)
    """Main field configuration."""

    appearance_dim: int = 16
    """Dimensionality of the appearance embedding."""
    use_temporal_appearance: bool = True
    """Whether to use temporal appearance or not."""
    temporal_appearance_freq: float = 1.0
    """Frequency of temporal appearance."""

    rgb_upsample_factor: int = 3
    """Upsampling factor for the rgb decoder."""
    rgb_hidden_dim: int = 32
    """Dimensionality of the hidden layers in the CNN."""

    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""

    compensate_upsampling_when_rendering: bool = True
    """Compensate for upsampling when asked to render an image of some given resolution."""
    normalize_depth: bool = False
    """Whether to normalize depth by dividing by accumulation."""
    verbose: bool = False
    """Whether to log additional images and out the model configuration."""

    @property
    def num_proposal_rounds(self):
        return len(self.sampling.num_proposal_samples)

    @property
    def fields(self) -> list[Union[NeuRADFieldConfig, NeuRADProposalFieldConfig]]:
        return [self.field, self.sampling.proposal_field_1, self.sampling.proposal_field_2]


class NeuRADModel(ADModel):
    """NeuRAD model.

    Args:
        config: NeuRAD configuration to instantiate model
    """

    config: NeuRADModelConfig
    def illuminate_vec(self, normals: Tensor, env: Tensor) -> Tensor:
        """Illuminate the vector with the environment map.
        Args:
            n: Normal vectors.
            env: Environment map.
        Returns:
            Tensor: Illuminated vectors.
        """
          # Ensure unit normals
        # normals = normals.expand(-1, 3)
        c1 = 0.282095
        c2 = 0.488603
        c3 = 1.092548
        c4 = 0.315392
        c5 = 0.546274

        c = env.unsqueeze(1)
        c = c.to('cuda')
        x, y, z = normals[..., 0, None], normals[..., 1, None], normals[..., 2, None]

        irradiance = (
            c[0] * c1 +
            c[1] * c2*y +
            c[2] * c2*z +
            c[3] * c2*x +
            c[4] * c3*x*y +
            c[5] * c3*y*z +
            c[6] * c4*(3*z*z-1) +
            c[7] * c3*x*z +
            c[8] * c5*(x*x-y*y)
        )
        return irradiance
    def env_rgb2gray(self):
        return self.env[:, 0] * 0.2989 + self.env[:, 1] * 0.5870 + self.env[:, 2] * 0.1140

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        if self.config.implementation == "tcnn" and not TCNN_EXISTS:
            print_tcnn_speed_warning("NeuRAD")
            self.config.implementation = "torch"
        self.field = self.config.field.setup(
            actors=self.dynamic_actors,
            static_scale=self.scene_box.aabb.max(),
            implementation=self.config.implementation,
        )
        self.env = torch.nn.Parameter(torch.rand(9, 3)*2 -1 ) 
        # self.env = torch.nn.Parameter(torch.uniform(-0.1, 0.1, size=(9, 3))  # ضرایب SH تصادفی

        # Appearance embedding settings
        dataset_metadata = self.kwargs["metadata"]
        self._duration = dataset_metadata["duration"]
        num_sensors = len(dataset_metadata["sensor_idx_to_name"])
        if self.config.use_temporal_appearance:
            self._num_embeds_per_sensor = math.ceil(self._duration * self.config.temporal_appearance_freq)
            num_embeds = num_sensors * self._num_embeds_per_sensor
        else:
            num_embeds = num_sensors
        self.appearance_embedding = torch.nn.Embedding(num_embeds, self.config.appearance_dim)
        self.fallback_sensor_idx = ViewerSlider("fallback sensor idx", 0, 0, num_sensors - 1, step=1)
        self.env_orginal = self.env
        self.env_view = ViewerDropdown("relighting", default_value= "orginal",options=["orginal","test1","test2","test3","test4"], cb_hook = self.env_view_update_callback)
        
        # Modality decoders
        hidden_dim = self.config.rgb_hidden_dim
        in_dim = self.config.field.nff_out_dim + self.config.appearance_dim
        self.rgb_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=self.config.rgb_upsample_factor,
                stride=self.config.rgb_upsample_factor,
            ),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )

        #ghazal added this line:
        hidden_dim = self.config.rgb_hidden_dim 
        in_dim = self.config.field.nff_out_dim + self.config.appearance_dim 
        self.albedo_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=self.config.rgb_upsample_factor,
                stride=self.config.rgb_upsample_factor,
            ),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )
        hidden_dim = self.config.rgb_hidden_dim 
        in_dim = self.config.field.nff_out_dim + self.config.appearance_dim + 9
        self.shadow_decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(inplace=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim,
                kernel_size=self.config.rgb_upsample_factor,
                stride=self.config.rgb_upsample_factor,
            ),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            BasicBlock(hidden_dim, hidden_dim, kernel_size=7, padding=3, use_bn=True),
            torch.nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0),
            torch.nn.Sigmoid(),
        )
        self.shadow_decoder.register_parameter("env", self.env)

        self.lidar_decoder = MLP(
            in_dim=self.config.field.nff_out_dim + self.config.appearance_dim,
            layer_width=32,
            out_dim=2,
            num_layers=3,
            implementation=self.config.implementation,
            out_activation=None,
        )

        # Sampler
        self.sampler = ProposalNetworkSampler(
            num_proposal_samples_per_ray=self.config.sampling.num_proposal_samples,
            num_nerf_samples_per_ray=self.config.sampling.num_nerf_samples,
            num_proposal_network_iterations=self.config.num_proposal_rounds,
            single_jitter=self.config.sampling.single_jitter,
            initial_sampler=PowerSampler(
                lambda_=self.config.sampling.power_lambda,
                scaling=self.config.sampling.power_scaling,
            ),
            update_sched=lambda x: 0,
        )
        self.proposal_fields = torch.nn.ModuleList(
            [
                conf.setup(
                    actors=self.dynamic_actors,
                    static_scale=self.scene_box.aabb.max(),
                    implementation=self.config.implementation,
                )
                for conf in (self.config.sampling.proposal_field_1, self.config.sampling.proposal_field_2)
            ]
        )
        self.density_fns = [lambda x: prop_field.get_density(x)[0] for prop_field in self.proposal_fields]

        # renderers
        self.renderer_feat = FeatureRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected") if self.config.normalize_depth else render_depth_simple
        self.renderer_normals = NormalsRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.shadow_loss = MSELoss()
        self.depth_loss = L1Loss(reduction="none")
        self.intensity_loss = MSELoss(reduction="none")
        self.vgg_loss = VGGPerceptualLossPix2Pix()
        self.ray_drop_loss = BCEWithLogitsLoss()
        self.interlevel_loss = zipnerf_interlevel_loss

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.median_l2 = lambda pred, gt: torch.median((pred - gt) ** 2)
        self.mean_rel_l2 = lambda pred, gt: torch.mean(((pred - gt) / gt) ** 2)
        self.rmse = lambda pred, gt: torch.sqrt(torch.mean((pred - gt) ** 2))
        self.chamfer_distance = lambda pred, gt: chamfer_distance(pred, gt, 1_000, True)

    def env_view_update_callback(self, viewer):
            if viewer.value == "orginal":
                self.env= self.env_orginal
            if viewer.value == "test1":
                self.env = torch.nn.Parameter(torch.tensor([
                            [3.250186443328857422e+00, 3.937959909439086914e+00, 3.989197015762329102e+00],
                            [5.101820230484008789e-01, 5.369477868080139160e-01, 5.093246698379516602e-01],
                            [-8.332927227020263672e-01, -9.154879450798034668e-01, -4.749252796173095703e-01],
                            [3.150682523846626282e-02, -3.080343455076217651e-02, 2.096310853958129883e-01],
                            [1.325290650129318237e-01, 1.582942903041839600e-01, 2.059240490198135376e-01],
                            [-1.453602015972137451e-01, -2.841771543025970459e-01, -2.341142147779464722e-01],
                            [1.737874299287796021e-01, 1.719687879085540771e-01, -2.212211489677429199e-02],
                            [1.232113316655158997e-01, 2.169368565082550049e-01, 1.488212347030639648e-01],
                            [2.533026635646820068e-01, 2.463037222623825073e-01, 2.009327411651611328e-01]
                        ], device='cuda:0', requires_grad=True))

            if viewer.value == "test2":   
                self.env = torch.nn.Parameter(torch.tensor([
                            [4.464160442352294922e+00, 5.662313461303710938e+00, 6.349784374237060547e+00],
                            [4.687011241912841797e-01, 5.115159153938293457e-01, 2.423315793275833130e-01],
                            [1.569044440984725952e-01, 6.701855361461639404e-02, 2.348522543907165527e-01],
                            [1.276165693998336792e-01, 2.658900618553161621e-01, 1.024356707930564880e-01],
                            [-2.676239004358649254e-03, 4.015997052192687988e-02, 4.235896468162536621e-02],
                            [2.558644674718379974e-02, 8.947437256574630737e-02, 8.677969872951507568e-02],
                            [5.453045964241027832e-01, 6.905375719070434570e-01, 4.189918637275695801e-01],
                            [1.169607192277908325e-01, 6.454331427812576294e-02, 1.568866521120071411e-01],
                            [6.923469156026840210e-02, 5.691013485193252563e-02, 9.221276640892028809e-02]
                        ], device='cuda:0', requires_grad=True))
            if viewer.value == "test3":   
                self.env = torch.nn.Parameter(torch.tensor([
                    [0.8, 0.5, 0.3], 
                    [0.6, 0.4, 0.2], 
                    [0.4, 0.3, 0.1],  
                    [0.3, 0.2, 0.1], 
                    [0.2, 0.1, 0.05], 
                    [0.1, 0.05, 0.02],
                    [0.5, 0.3, 0.2], 
                    [0.4, 0.2, 0.1],  
                    [0.3, 0.15, 0.05] 
                ], device='cuda:0', requires_grad=True))
            if viewer.value == "test4":   
                self.env = torch.nn.Parameter(torch.tensor([[ 3.8766,  6.0797,  6.3498],
                    [ 0.4153,  0.5558,  0.2423],
                    [ 0.1494,  0.0823,  0.2349],
                    [ 0.1004,  0.2773,  0.1024],
                    [-0.0067,  0.0397,  0.0424],
                    [ 0.0165,  0.0916,  0.0868],
                    [ 0.4736,  0.7415,  0.4190],
                    [ 0.1099,  0.0759,  0.1569],
                    [ 0.0632,  0.0635,  0.0922]], device='cuda:0', requires_grad=True))

    @property
    def fields(self) -> list[Union[NeuRADField, NeuRADProposalField]]:
        """Convenience accessor for all fields."""
        return [self.field, *self.proposal_fields]  # type: ignore

    def disable_ray_drop(self):
        self.config.loss.ray_drop_loss_mult = 0.0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the parameter groups for the optimizer."""
        param_groups = super().get_param_groups()
        for field_ in self.fields:
            field_.get_param_groups(param_groups)
        param_groups["fields"] += list(self.lidar_decoder.parameters())
        param_groups["fields"] += list(self.appearance_embedding.parameters())
        param_groups["cnn"] += self.rgb_decoder.parameters()
        param_groups["cnn"] += list(self.albedo_decoder.parameters())
        param_groups["cnn"] += list(self.shadow_decoder.parameters())
        param_groups["cnn"] += [self.env]
       
        return param_groups

    def get_training_callbacks(self, _) -> List[TrainingCallback]:
        callbacks = []
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.sampler.step_cb,
            )
        )
        return callbacks

    def forward(
        self,
        ray_bundle: RayBundle,
        patch_size: Tuple[int, int] = (1, 1),
        intensity_for_cam: bool = False,
        calc_lidar_losses: bool = True,
    ):
        return self.get_outputs(ray_bundle, patch_size, intensity_for_cam, calc_lidar_losses)

    def get_outputs(
        self,
        ray_bundle: RayBundle,
        patch_size: Tuple[int, int],
        intensity_for_cam: bool = False,
        calc_lidar_losses: bool = True,
    ):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        nff_outputs = self.get_nff_outputs(ray_bundle, calc_lidar_losses)
        rgb, intensity, ray_drop_logits, shadow,rgb_org,albedo = self.decode_features(
            features=nff_outputs["features"],
            patch_size=patch_size,
            is_lidar=ray_bundle.metadata.get("is_lidar"),
            intensity_for_cam=intensity_for_cam,
            normals = nff_outputs["normals"],
            ray_bundle=ray_bundle,
        )
       # Drop features because it's too trippy (and big)
        nff_outputs.pop("features", None)
        if rgb is not None:
            nff_outputs["rgb"] = rgb
        if rgb is not None:
            nff_outputs["rgb_org"] = rgb_org
        if shadow is not None:
            nff_outputs["shadow"] = shadow
        if albedo is not None:
            nff_outputs["albedo"] = albedo
        if intensity is not None:
            nff_outputs["intensity"] = intensity.float()
        if ray_drop_logits is not None:
            nff_outputs["ray_drop_logits"] = ray_drop_logits.float()
        # nff_outputs["sdf"] = nff_outputs["sdf"].squeeze(-1)[~ray_bundle.metadata.get("is_lidar")[..., 0]]
        return nff_outputs

    def decode_features(
        self,
        features: Tensor,
        patch_size: Tuple[int, int],
        is_lidar: Optional[Tensor] = None,
        intensity_for_cam: bool = False,
        normals = None,
        ray_bundle=RayBundle,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Take the neural feature field feature, and render them to rgb and/or intensity."""
        if is_lidar is None:
            lidar_features, cam_features = torch.empty(0), features
            lidar_normals, cam_normals = torch.empty(0), normals
            camera_ray_bundle = ray_bundle
        else:
            lidar_features, cam_features = features[is_lidar[..., 0]], features[~is_lidar[..., 0]]
            lidar_normals, cam_normals = normals[is_lidar[..., 0]], normals[~is_lidar[..., 0]]
            camera_ray_bundle = ray_bundle[~is_lidar[..., 0]]
        # Decode lidar features
        if intensity_for_cam:  # Useful for visualization
            intensity, ray_drop_logit = self.lidar_decoder(features).float().split(1, dim=-1)  # TODO: hmm
        elif lidar_features.numel() > 0:
            intensity, ray_drop_logit = self.lidar_decoder(lidar_features).split(1, dim=-1)
        else:
            intensity, ray_drop_logit = None, None
        intensity = intensity.sigmoid() if intensity is not None else None

        if cam_features.numel() == 0:
            rgb1 = None
        H,W =patch_size
        # need to reshape the ray features to apply the cnn decoder
        cam_feature_patches = cam_features.view(-1, *patch_size, cam_features.shape[-1])  # B x D x D x C
        cam_feature_patches = cam_feature_patches.permute(0, 3, 1, 2)  # B x C x D x D
        
        env_grayscale = self.env_rgb2gray()
        env_gray_reshaped = env_grayscale.view(1, 9, 1, 1)
        env_gray_broadcasted = env_gray_reshaped.expand(cam_feature_patches.size(0), 9, cam_feature_patches.size(2), cam_feature_patches.size(3))
        env_gray_broadcasted = env_gray_broadcasted.to('cuda')
        input_shadow_concate = torch.cat((cam_feature_patches, env_gray_broadcasted), dim=1)
        shadow = self.shadow_decoder(input_shadow_concate)
        albedo =  self.albedo_decoder(cam_feature_patches)
        irradiance1 = self.illuminate_vec(cam_normals.view(-1, H, W, 3), self.env)
        irradiance1 = torch.relu(irradiance1)  # Adjust max value as appropriate
        irradiance = irradiance1 ** (1 / 2.2)
        
        # Interpolate with better control
        irr = F.interpolate(
            irradiance.view(-1, *patch_size, irradiance.shape[-1]).permute(0, 3, 1, 2),
            size=(albedo.shape[-2], albedo.shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        fg_pure_rgb_map = irr * albedo
        rgb1 = fg_pure_rgb_map * shadow
        rgb1 = torch.clamp(rgb1, 0, 1)

        rgb_org = self.rgb_decoder(cam_feature_patches)  # B x 3 x upsample x upsample
        rgb1 = rgb1.permute(0, 2, 3, 1)  # B x upsample x upsample x 3
        rgb_org = rgb_org.permute(0, 2, 3, 1) # B x upsample x upsample x 3
        shadow1 = shadow.permute(0, 2, 3, 1)
        albedo1 = albedo.permute(0, 2, 3, 1)  # B x upsample x upsample x 3
        #rgb1 = rgb1.clamp(0, 1)  # Ensure final RGB values are within the valid range
        return rgb1, intensity, ray_drop_logit, shadow1, rgb_org, albedo1


    def get_nff_outputs(self, ray_bundle: RayBundle, calc_lidar_losses: bool = False) -> Dict[str, Tensor]:
        """Run the neural feature field, and return the rendered outputs."""
        self._scale_pixel_area(ray_bundle)
        ray_samples, proposal_ray_samples, proposal_weights = self._get_ray_samples(ray_bundle)

        outputs = self.field(ray_samples)

        # Render rays
        nff_outputs = {}
        weights = self._render_weights(outputs, ray_samples)
        accumulation = self.renderer_accumulation(weights=weights[..., None])

        # Put remaining accumulation on last (sky) sample and render features
        weights = torch.cat((weights[..., :-1], weights[..., -1:] + 1 - accumulation), dim=-1).unsqueeze(-1)
        

        with torch.enable_grad():
            ray_samples_ = copy.deepcopy(ray_samples)
            outputs_ = self.field(ray_samples_)
            inputs = outputs_['gas'].mean
            output_sdf_ = outputs_[FieldHeadNames.SDF]
            normals = torch.autograd.grad(
                inputs=inputs,
                outputs=output_sdf_,
                grad_outputs=torch.ones_like(output_sdf_, requires_grad=False),
            )[0]
        normals = normals.squeeze(2)
        normals = normals.detach()
        normals = self.renderer_normals(normals, weights=weights.detach(), normalize=True)

        features = self.renderer_feat(features=outputs[FieldHeadNames.FEATURE], weights=weights)
        if self.config.appearance_dim > 0:  # add appearance embedding
            appearance = self._get_appearance_embedding(ray_bundle, features)
            features = torch.cat([features, appearance], dim=-1)
        # Discard sky sample for all remaining purposes
        weights, ray_samples = weights[..., :-1, :], ray_samples[..., :-1]
        outputs = {k: v[..., :-1, :] for k, v in outputs.items()}
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        nff_outputs = {
            "features": features,
            "depth": depth,
            "accumulation": accumulation,
            "normals": normals,
        }
        # if not self.training and FieldHeadNames.NORMALS in outputs:
        #     nff_outputs["normals"] = self.renderer_normals(normals=outputs[FieldHeadNames.NORMALS], weights=weights)

        # Handle proposal outputs
        for i, (prop_w, prop_rs) in enumerate(zip(proposal_weights, proposal_ray_samples)):
            nff_outputs[f"prop_depth_{i}"] = self.renderer_depth(prop_w, prop_rs)
            if self.training and calc_lidar_losses:
                weights_mask = (~prop_rs.metadata["is_close_to_lidar"]) & prop_rs.metadata["is_lidar"]
                nff_outputs[f"prop_weights_loss_{i}"] = ((prop_w * weights_mask) ** 2).sum()
        if self.training:
            # Need to render the static samples separately for proposal supervision
            nff_outputs["weights_list"] = proposal_weights + [weights]
            nff_outputs["ray_samples_list"] = proposal_ray_samples + [ray_samples]

        if self.training and calc_lidar_losses:
            assert (metadata := ray_samples.metadata) is not None
            weights_mask = ((~metadata["is_close_to_lidar"]) & metadata["is_lidar"]).squeeze(-1)
            weights_idx = weights_mask.nonzero(as_tuple=True)
            nff_outputs["non_nearby_weights"] = weights[weights_idx]
            # TODO: get rid of this
            ray_indices = torch.arange(weights.shape[0], device=weights.device).unsqueeze(-1)
            ray_indices = ray_indices.repeat(1, *weights.shape[1:-1])
            lidar_start_ray = ray_bundle.metadata["is_lidar"].int().argmax()  # argmax gives first True
            nff_outputs["non_nearby_lidar_ray_indices"] = ray_indices[weights_idx] - lidar_start_ray

        return nff_outputs

    def _get_appearance_embedding(self, ray_bundle, features):
        sensor_idx = ray_bundle.metadata.get("sensor_idxs")
        if sensor_idx is None:
            assert not self.training, "Sensor sensor_idx must be present in metadata during training"
            sensor_idx = torch.full_like(features[..., :1], self.fallback_sensor_idx.value, dtype=torch.long)

        if self.config.use_temporal_appearance:
            time_idx = ray_bundle.times / self._duration * (embd_per_sensor := self._num_embeds_per_sensor)
            before_idx = time_idx.floor().clamp(0, embd_per_sensor - 1)
            after_idx = (before_idx + 1).clamp(0, embd_per_sensor - 1)
            ratio = time_idx - before_idx
            # unwrap to true embedding indices, which also account for the sensor index, not just the time index
            before_idx, after_idx = (x + sensor_idx * embd_per_sensor for x in (before_idx, after_idx))
            before_embed = self.appearance_embedding(before_idx.squeeze(-1).long())
            after_embed = self.appearance_embedding(after_idx.squeeze(-1).long())
            embed = before_embed * (1 - ratio) + after_embed * ratio
        else:
            embed = self.appearance_embedding(sensor_idx.squeeze(-1))
        return embed

    def _get_ray_samples(self, ray_bundle: RayBundle) -> tuple[RaySamples, List[RaySamples], List[Tensor]]:
        # Sampling
        if ray_bundle.fars is not None:
            ray_bundle.fars.clamp_max_(sky_distance := self.config.sampling.sky_distance)
        else:
            ray_bundle.fars = torch.full_like(ray_bundle.pixel_area, sky_distance := self.config.sampling.sky_distance)
        ray_bundle.nears = ray_bundle.nears if ray_bundle.nears is not None else torch.zeros_like(ray_bundle.fars)
        ray_samples, prop_weights, prop_ray_samples = self.sampler(ray_bundle, self.density_fns, pass_ray_samples=True)
        # Efficient "sky-field" by setting last sample to end extremely far away
        dist_to_sky = sky_distance - ray_samples.frustums.ends[..., -1, 0]
        ray_samples.frustums.ends[..., -1, 0] += dist_to_sky
        ray_samples.deltas[..., -1, 0] += dist_to_sky
        ray_samples.spacing_ends[..., -1, 0] = 1 - EPS  # Hacky, but sky is ish at infinity

        if self.training and "is_lidar" in ray_bundle.metadata:
            self._compute_is_close_to_lidar(ray_samples, *prop_ray_samples)
        return ray_samples, prop_ray_samples, prop_weights

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        if "image" in batch:
            image, rgb = batch["image"].to(self.device), outputs["rgb"]
            metrics_dict["psnr"] = self.psnr(rgb.detach(), image)
        if "lidar" in batch:
            is_lidar = batch["is_lidar"][:, 0].to(self.device)
            n_lidar_rays = is_lidar.sum()
            did_return = batch["did_return"][batch["is_lidar"].squeeze(-1)].squeeze(-1).to(self.device)
            points_intensities = batch["lidar"][..., 3:4].to(self.device)
            termination_depth = batch["distance"].to(self.device)

            pred_depth = outputs["depth"][is_lidar]
            ray_drop_logits = outputs["ray_drop_logits"]
            pred_intensity = outputs["intensity"]

            # eval metrics
            metrics_dict["depth_median_l2"] = self.median_l2(pred_depth[did_return], termination_depth[did_return])
            metrics_dict["depth_mean_rel_l2"] = self.mean_rel_l2(pred_depth[did_return], termination_depth[did_return])
            metrics_dict["intensity_rmse"] = self.rmse(pred_intensity[did_return], points_intensities[did_return])
            metrics_dict["ray_drop_accuracy"] = (
                ((ray_drop_logits.sigmoid() > 0.5).squeeze(-1) == ~did_return).float().mean()
            )
 
            # train metrics / losses
            if self.training:
                # Adjust target depth for non-returning rays
                nonret_lid_dist = torch.tensor(
                    self.config.loss.non_return_lidar_distance, device=termination_depth.device
                )
                target_depth = termination_depth.clone()
                target_depth[~did_return] = pred_depth.detach()[~did_return].maximum(nonret_lid_dist)
                unreduced_depth_loss = self.depth_loss(target_depth, pred_depth)
                unreduced_depth_loss[~did_return] *= self.config.loss.non_return_loss_mult
                # TODO: get rid of quantile mask
                quantile = torch.quantile(unreduced_depth_loss, self.config.loss.quantile_threshold)
                quantile_mask = (unreduced_depth_loss < quantile).squeeze(-1)

                metrics_dict["depth_loss"] = torch.mean(unreduced_depth_loss[quantile_mask])

                quant_and_return = quantile_mask & did_return
                metrics_dict["intensity_loss"] = self.intensity_loss(
                    points_intensities[quant_and_return], outputs["intensity"][quant_and_return]
                ).mean()

                metrics_dict["ray_drop_loss"] = self.ray_drop_loss(
                    ray_drop_logits, (~did_return).unsqueeze(-1).to(ray_drop_logits)
                )
                # quantile_weights_mask = quantile_mask[outputs["non_nearby_lidar_ray_indices"]].squeeze(-1)
                weights_loss = (outputs["non_nearby_weights"] ** 2).sum()
                metrics_dict["carving_loss"] = weights_loss / n_lidar_rays  # avg per ray

                # Lidar proposal losses
                for prop_i in range(self.config.num_proposal_rounds):
                    pred_depth = outputs[f"prop_depth_{prop_i}"][is_lidar]
                    target_depth = termination_depth.clone()
                    target_depth[~did_return] = pred_depth.detach()[~did_return].maximum(nonret_lid_dist)
                    unreduced_depth_loss = self.depth_loss(target_depth, pred_depth)
                    unreduced_depth_loss[~did_return] *= self.config.loss.non_return_loss_mult
                    metrics_dict[f"depth_loss_{prop_i}"] = torch.mean(unreduced_depth_loss)
                    metrics_dict[f"carving_loss_{prop_i}"] = outputs[f"prop_weights_loss_{prop_i}"] / n_lidar_rays

        if self.training and "weights_list" in outputs:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        if self.config.field.use_sdf:
            metrics_dict["sdf_to_density"] = float(self.field.sdf_to_density.beta)
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        conf = self.config.loss
        loss_dict = {}
        if "image" in batch:
            image, rgb, rgb_org = batch["image"].to(self.device), outputs["rgb"], outputs["rgb_org"]
            loss_dict["rgb_loss"] = self.rgb_loss(image, rgb) * conf.rgb_mult
            loss_dict["rgb_org_loss"] = self.rgb_loss(image, rgb_org) * conf.rgb_mult
            loss_dict["rgb_org_loss"] += self.rgb_loss(rgb_org.detach(), rgb) * (conf.rgb_mult/2)
            if conf.vgg_mult > 0.0:
                loss_dict["vgg_loss"] = self.vgg_loss(rgb, image) * conf.vgg_mult
                loss_dict["vgg_org_loss"] = self.vgg_loss( rgb_org, image) * conf.rgb_mult
                loss_dict["vgg_org_loss"] += self.vgg_loss(rgb, rgb_org.detach()) * conf.vgg_mult


        shadow = outputs["shadow"]
        loss_dict["shadow_loss"] = self.shadow_loss(torch.ones_like(shadow), shadow) * conf.shadow_mult
        
        if self.training:
            if "weights_list" in outputs:
                loss_dict["interlevel_loss"] = self.config.loss.interlevel_loss_mult * self.interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )
                assert metrics_dict is not None and "distortion" in metrics_dict
                loss_dict["distortion_loss"] = self.config.loss.distortion_loss_mult * metrics_dict["distortion"]
                prop_depth_mult = conf.prop_lidar_loss_mult * conf.depth_mult
                prop_carv_mult = conf.prop_lidar_loss_mult * conf.carving_mult
                for i_prop in range(self.config.num_proposal_rounds):
                    loss_dict[f"depth_loss_{i_prop}"] = prop_depth_mult * metrics_dict[f"depth_loss_{i_prop}"]
                    loss_dict[f"carving_loss_{i_prop}"] = prop_carv_mult * metrics_dict[f"carving_loss_{i_prop}"]
            assert metrics_dict
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = conf.depth_mult * metrics_dict["depth_loss"]
            if "intensity_loss" in metrics_dict:
                loss_dict["intensity_loss"] = conf.intensity_mult * metrics_dict["intensity_loss"]
            if "carving_loss" in metrics_dict:
                loss_dict["carving_loss"] = conf.carving_mult * metrics_dict["carving_loss"]
            if "ray_drop_loss" in metrics_dict:
                loss_dict["ray_drop_loss"] = conf.ray_drop_loss_mult * metrics_dict["ray_drop_loss"]
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict = {}
        images_dict = {}
        if "image" in batch:
            image, rgb = batch["image"].to(self.device), outputs["rgb"]
            images_dict["img"] = torch.cat([image, rgb], dim=1)
            images_dict["depth"] = colormaps.apply_depth_colormap(outputs["depth"])
            if self.config.verbose:
                images_dict["accumulation"] = colormaps.apply_colormap(outputs["accumulation"])
                # Add proposal depthmaps
                for i in range(self.config.num_proposal_rounds):
                    key = f"prop_depth_{i}"
                    prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
                    images_dict[key] = prop_depth_i

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            image = torch.moveaxis(image, -1, 0)[None, ...]
            rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

            # all of these metrics will be logged as scalars
            metrics_dict["psnr"] = float(self.psnr(image, rgb))
            metrics_dict["ssim"] = float(self.ssim(image, rgb))  # type: ignore
            metrics_dict["lpips"] = float(self.lpips(image, rgb))

        if "lidar" in batch:
            points = batch["lidar"].to(self.device)
            if "is_lidar" not in batch:
                batch["is_lidar"] = torch.ones(*batch["lidar"].shape[:-1], 1, dtype=torch.bool, device=self.device)
            if "did_return" not in batch:
                batch["did_return"] = torch.ones(*batch["lidar"].shape[:-1], 1, dtype=torch.bool, device=self.device)

            ray_drop_logits = outputs["ray_drop_logits"]
            pred_depth = outputs["depth"]
            did_return = batch["did_return"][:, 0].to(self.device)
            is_lidar = batch["is_lidar"][:, 0].to(self.device)
            metrics_dict["depth_median_l2"] = float(
                self.median_l2(pred_depth[is_lidar][did_return], batch["distance"][did_return])
            )
            metrics_dict["depth_mean_rel_l2"] = float(
                self.mean_rel_l2(pred_depth[is_lidar][did_return], batch["distance"][did_return])
            )
            metrics_dict["intensity_rmse"] = float(self.rmse(outputs["intensity"][did_return], points[did_return, 3:4]))
            metrics_dict["ray_drop_accuracy"] = float(
                ((ray_drop_logits.sigmoid() > 0.5).squeeze(-1) == ~did_return).float().mean()
            )
            if self.config.loss.ray_drop_loss_mult > 0.0:
                pred_points_did_return = (ray_drop_logits.sigmoid() < 0.5).squeeze(-1)
            else:
                pred_points_did_return = (pred_depth < self.config.loss.non_return_lidar_distance).squeeze(-1)
            if pred_points_did_return.any() and points.shape[0] > 0 and did_return.any():
                pred_points = outputs["points"][is_lidar][pred_points_did_return]
                metrics_dict["chamfer_distance"] = float(
                    self.chamfer_distance(pred_points[..., :3], points[did_return, :3])
                )
            else:
                metrics_dict["chamfer_distance"] = points[did_return, :3].norm(dim=-1).mean()
        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        if len(camera_ray_bundle.shape) == 1:  # lidar
            output_size, patch_size = (camera_ray_bundle.shape[0],), (1, 1)
            is_lidar = torch.ones_like(camera_ray_bundle.pixel_area, dtype=torch.bool)
        else:  # camera
            assert len(camera_ray_bundle.shape) == 2, "Raybundle should be 2d (an image/patch)"
            if self.config.compensate_upsampling_when_rendering:
                # shoot rays at a lower resolution and upsample the output to the target resolution
                step = self.config.rgb_upsample_factor
                camera_ray_bundle = camera_ray_bundle[step // 2 :: step, step // 2 :: step]
            output_size = patch_size = (camera_ray_bundle.shape[0], camera_ray_bundle.shape[1])
            camera_ray_bundle = camera_ray_bundle.reshape((-1,))
            is_lidar = None

        # Run chunked forward pass through NFF only
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.get_nff_outputs(ray_bundle, calc_lidar_losses=False)
            
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(*output_size, -1)  # type: ignore

        features = outputs["features"].view(-1, outputs["features"].shape[-1])
        rgb, intensity, ray_drop_logit, shadow,rgb_org,albedo= self.decode_features(
            features, patch_size=patch_size, is_lidar=is_lidar, intensity_for_cam=True, normals = outputs["normals"],ray_bundle = camera_ray_bundle )
       
        if rgb is not None:
            outputs["rgb"] = rgb.squeeze(0)
        if rgb_org is not None:
            outputs["rgb_org"] = rgb_org.squeeze(0)
        if shadow is not None:
            outputs["shadow"] = shadow.squeeze(0)
        if albedo is not None:
            outputs["albedo"] = albedo.squeeze(0)
        if intensity is not None:
            outputs["intensity"] = intensity.view(*output_size, -1)
        if ray_drop_logit is not None:
            outputs["ray_drop_logits"] = ray_drop_logit.view(*output_size, -1)
            outputs["ray_drop_prob"] = ray_drop_logit.view(*output_size, -1).sigmoid()
        # outputs.pop('normals')

        return outputs

    def _compute_is_close_to_lidar(self, *all_ray_samples: RaySamples) -> None:
        """Compute which rays are close to the lidar."""
        for ray_samples in all_ray_samples:
            if ray_samples is None:
                continue
            assert ray_samples.metadata is not None
            metadata, frustums = ray_samples.metadata, ray_samples.frustums
            sample_distance = (frustums.starts + frustums.ends) * 0.5
            mask = metadata["is_lidar"].clone()  # handle that is_lidar is expanded
            idx = mask.nonzero(as_tuple=True)
            sample_distance = sample_distance[idx]
            # directions_norm, in case of lidar, is the distance
            dist = metadata["directions_norm"][idx] - sample_distance
            # same as mask[mask] since it will write to itself in place and break
            close_to_hit = dist.abs() < self.config.loss.carving_epsilon
            if "did_return" in metadata.keys():
                did_return = metadata["did_return"][idx]
                in_lidar_range = sample_distance < self.config.loss.non_return_lidar_distance
                mask[idx] = (did_return & close_to_hit) | (
                    (~did_return) & in_lidar_range
                )  # TODO: should this be in_range or out_of_range?
            else:
                mask[idx] = close_to_hit
            metadata["is_close_to_lidar"] = mask

    def _scale_pixel_area(self, ray_bundle: RayBundle):
        is_lidar = ray_bundle.metadata.get("is_lidar")
        scaling = torch.ones_like(ray_bundle.pixel_area)
        if is_lidar is not None:
            scaling[~is_lidar] = self.config.rgb_upsample_factor**2
        else:
            scaling = self.config.rgb_upsample_factor**2
        ray_bundle.pixel_area = ray_bundle.pixel_area * scaling

    def _render_weights(self, outputs, ray_samples):
        value = outputs[FieldHeadNames.ALPHA if self.config.field.use_sdf else FieldHeadNames.DENSITY].squeeze(-1)
        if self.device.type in ("cpu", "mps"):
            # Note: for debugging on devices without cuda
            weights = torch.zeros_like(value) + 0.5
        elif self.config.field.use_sdf:
            weights, _ = nerfacc.render_weight_from_alpha(value)
        else:
            weights, _, _ = nerfacc.render_weight_from_density(
                t_ends=ray_samples.frustums.ends.squeeze(-1),
                t_starts=ray_samples.frustums.starts.squeeze(-1),
                sigmas=value,
            )
        return weights


def render_depth_simple(
    weights: Float[Tensor, "*batch num_samples 1"],
    ray_samples: RaySamples,
    ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
    num_rays: Optional[int] = None,
) -> Float[Tensor, "*batch 1"]:
    steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    return nerfacc.accumulate_along_rays(weights[..., 0], values=steps, ray_indices=ray_indices, n_rays=num_rays)
