import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2

from .Stimulus import Stimulus

class Bar(nn.Module, Stimulus):
    """
    A class to generate a bar stimulus.
    """
    def __init__(self, canvas_size, location, length, width, contrast, orientation, grey_level,
                 pixel_boundaries=None):
        """
        Args:
            canvas_size (int): The canvas size [width, height].
            location (list or Parameter): specifies the center position of the bar. Can be either of type list or an
                object from parameters.py module. This module has 3 relevant classes: FiniteParameter, FiniteSelection,
                and UniformRange. FiniteParameter objects will be treated exactly like lists. FiniteSelection objects
                will generate n samples from the given list of values from a probability mass function. UniformRange
                objects will sample from a continuous distribution within the defined parameter ranges. If location is
                of type UniformRange, there cannot be an additional argument for the cumulative density distribution.
            length (float or Parameter): determines the bar length. Can be either of type list or an object from
                parameters.py module.
            width (float or Parameter): determines the bar width. Can be either of type list or an object from parameters.py
                module.
            contrast (float or Parameter): defines the amplitude of the stimulus in %. Takes values from -1 to 1. E.g., for
                a grey_level=-0.2 and pixel_boundaries=[-1,1], a contrast of 1 (=100%) means the amplitude of the bar
                stimulus is 0.8. Negative contrasts lead to a white bar on less luminated background. Can be either of
                type list or an object from parameters.py module.
            orientation (float or Parameter): determines the orientation of a bar. Its values are given in [rad] and can
                range from [0, pi). Can be either of type list or an object from parameters.py module.
            grey_level (float or Parameter): determines the mean luminance (pixel value) of the image. Can be either of type
                list or an object from parameters.py module.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].
        """
        super().__init__()

        self._params_dict = dict(
            location=location,
            length=length,
            width=width,
            contrast=contrast,
            orientation=orientation,
            grey_level=grey_level
        )

        location = self.parse_parameter(location),
        length = self.parse_parameter(length),
        width = self.parse_parameter(width),
        contrast = self.parse_parameter(contrast),
        orientation = self.parse_parameter(orientation),
        grey_level = self.parse_parameter(grey_level)

        self.location = nn.Parameter(torch.Tensor(location))
        self.length = nn.Parameter(torch.Tensor([length]))
        self.width = nn.Parameter(torch.Tensor([width]))
        self.contrast = nn.Parameter(torch.Tensor([contrast]))
        self.orientation = nn.Parameter(torch.Tensor([orientation]))
        self.grey_level = nn.Parameter(torch.Tensor([grey_level]))

        self.canvas_size = canvas_size
        self.pixel_boundaries = [-1, 1] if pixel_boundaries is None else pixel_boundaries

    def get_config(self):
        return dict(
            location=self.location.detach().numpy().tolist(),
            length=self.length.item(),
            width=self.width.item(),
            contrast=self.contrast.item(),
            orientation=self.orientation.item(),
            grey_level=self.grey_level.item()
        )

    def forward(self):
        return self.bar(
            canvas_size = self.canvas_size,
            location = self.location,
            length = self.length,
            width = self.width,
            contrast = self.contrast,
            orientation = self.orientation,
            grey_level = self.grey_level,
            pixel_boundaries = self.pixel_boundaries
        )

    @staticmethod
    def bar(canvas_size, location, length, width, contrast, orientation, grey_level, pixel_boundaries):
        # if width > length:
        #     raise ValueError("width cannot be larger than length.")

        # coordinate grid
        x, y = torch.meshgrid(
            torch.arange(canvas_size[0]) - location[0],
            torch.arange(canvas_size[1]) - location[1]
        )
        coords = torch.stack([x.flatten(), y.flatten()])

        # rotation matrix
        orientation = torch.pi - orientation
        R = torch.Tensor([
            [torch.cos(orientation), -torch.sin(orientation)],
            [torch.sin(orientation), torch.cos(orientation)]
        ])

        # scaling matrix
        A = torch.Tensor([
            [width / 2, 0],
            [0, length / 2]
        ])

        # inverse base change
        Minv = torch.inverse(R @ A) @ (coords)

        # infinity norm with "radius" 1 will induce desired rectangle
        Minv_norm = torch.max(torch.abs(Minv), dim=0).values

        M_inv_norm_mat = torch.reshape(Minv_norm, x.shape)
        bar_no_contrast = M_inv_norm_mat <= 1

        # add contrast
        amplitude = contrast * min(
            abs(pixel_boundaries[0] - grey_level),
            abs(pixel_boundaries[1] - grey_level)
        )

        bar = amplitude * bar_no_contrast + grey_level

        return bar.view((1, 1, *canvas_size)).cuda()

