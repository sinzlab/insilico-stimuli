import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

from .Stimulus import Stimulus
from .CenterSurround import CenterSurround

class PlaidsGrating(nn.Module, Stimulus):
    """
    A class to generate a Plaids Grating stimulus.
    """
    def __init__(self, canvas_size, location, size_total, contrast_preferred, contrast_overlap, spatial_frequency,
                 orientation, phase, grey_level, angle=None, pixel_boundaries=None):
        """
        Params:
            canvas_size (int): The canvas size [width, height].
            location (list or Parameter): specifies the center position of the DoG. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactly like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            size (float or Parameter): Standard deviation of the center Gaussian. Can be either a list or an object from
                parameters.py module.
            size_scale_surround (float or Parameter): Scaling factor defining how much larger the standard deviation of the
                surround Gaussian is relative to the size of the center Gaussian. Must have values larger than 1. Can be
                either a list or an object from parameters.py module.
            contrast (float or Parameter): Contrast of the center Gaussian in %. Takes values from -1 to 1. Can be either a
                list or an object from parameters.py module. Negative contrast yield to inverted stimuli (Gaussian in
                the center is negative, while the surround one is positive).
            contrast_scale_surround (float or Parameter): Contrast of the surround Gaussian relative to the center Gaussian.
                Should be between 0 and 1. Can be either a list or an object from parameters.py module.
            grey_level (float or Parameter): The mean luminance/pixel value. Can be either a list or an object from
                parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
        Returns:
            2D torch.tensor: A plaids grating stimulus.
        """

        super().__init__()

        self._params_dict = dict(
            location=location,
            size_total=size_total,
            contrast_preferred=contrast_preferred,
            contrast_overlap=contrast_overlap,
            spatial_frequency=spatial_frequency,
            orientation=orientation,
            phase=phase,
            grey_level=grey_level,
            angle=angle
        )

        location = self.parse_parameter(location)
        size_total = self.parse_parameter(size_total)
        contrast_preferred=self.parse_parameter(contrast_preferred)
        contrast_overlap=self.parse_parameter(contrast_overlap)
        spatial_frequency=self.parse_parameter(spatial_frequency)
        orientation=self.parse_parameter(orientation)
        phase=self.parse_parameter(phase)
        grey_level=self.parse_parameter(grey_level)
        angle=self.parse_parameter(angle)

        self.location = nn.Parameter(torch.Tensor(location))
        self.size_total = nn.Parameter(torch.Tensor([size_total]))
        self.contrast_preferred = nn.Parameter(torch.Tensor([contrast_preferred]))
        self.contrast_overlap = nn.Parameter(torch.Tensor([contrast_overlap]))
        self.spatial_frequency = nn.Parameter(torch.Tensor([spatial_frequency]))
        self.orientation = nn.Parameter(torch.Tensor([orientation]))
        self.phase = nn.Parameter(torch.Tensor([phase]))
        self.grey_level = nn.Parameter(torch.Tensor([grey_level]))
        self.angle = nn.Parameter(torch.Tensor([angle]))

        self.canvas_size = canvas_size
        self.pixel_boundaries = [-1, 1] if pixel_boundaries is None else pixel_boundaries

    def get_config(self):
        return dict(
            location=self.location.detach().numpy().tolist(),
            size_total=self.size_total.item(),
            contrast_preferred=self.contrast_preferred.item(),
            contrast_overlap=self.contrast_overlap.item(),
            spatial_frequency=self.spatial_frequency.item(),
            orientation=self.orientation.item(),
            phase=self.phase.item(),
            grey_level=self.grey_level.item(),
            angle=self.angle.item()
        )

    def forward(self):
        circular_grating_preferred = CenterSurround(
            self.canvas_size,
            location=self.location,
            size_total=self.size_total,
            size_center=1,
            size_surround=1,
            contrast_center=self.contrast_overlap,
            contrast_surround=0,
            orientation_center=self.orientation,
            orientation_surround=0,
            spatial_frequency_center=self.spatial_frequency,
            spatial_frequency_surround=0,
            phase_center=self.phase,
            phase_surround=0,
            grey_level=self.grey_level,
            pixel_boundaries=self.pixel_boundaries
        )

        circular_grating_overlap = CenterSurround(
            self.canvas_size,
            location=self.location,
            size_total=self.size_total,
            size_center=1,
            size_surround=1,
            contrast_center=self.contrast_overlap,
            contrast_surround=0,
            orientation_center=self.orientation + self.angle,
            orientation_surround=0,
            spatial_frequency_center=self.spatial_frequency,
            spatial_frequency_surround=0,
            phase_center=self.phase,
            phase_surround=0,
            grey_level=self.grey_level,
            pixel_boundaries=self.pixel_boundaries
        )

        plaid = circular_grating_preferred() + circular_grating_overlap()

        return plaid
