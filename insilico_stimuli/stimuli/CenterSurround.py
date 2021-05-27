import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

from .Stimulus import Stimulus


class CenterSurround(nn.Module, Stimulus):
    """
        A class to generate a center surround stimulus.
    """
    def __init__(self, canvas_size, location, size_total, size_center, size_surround, contrast_center,
                 contrast_surround, orientation_center, orientation_surround, spatial_frequency_center,
                 phase_center, grey_level, spatial_frequency_surround=None, phase_surround=None,
                 pixel_boundaries=None):
        """
        Args:
            canvas_size (tuple of integers): Image height and width.
            location (list of float or Parameter): The center position of the Center-Surround stimulus.
            size_total (float or Parameter): The overall size of the Center-Surround stimulus as the radius in [number of pixels].
            size_center (float or Parameter): The size of the center as a fraction of the overall size.
            size_surround (float or Parameter): The size of the surround as a fraction of the overall size.
            contrast_center (float or Parameter): The contrast of the center grating in %. Takes values from 0 to 1.
            contrast_surround (float or Parameter): The contrast of the surround grating in %. Takes values from 0 to 1.
            orientation_center (float or Parameter): The orientation of the center grating.
            orientation_surround (float or Parameter): The orientation of the surround grating.
            spatial_frequency_center (float or Parameter): The inverse of the wavelength of the center gratings.
            spatial_frequency_surround (float or Parameter): The inverse of the wavelength of the surround gratings.
            phase_center (float or Parameter): The cosine phase-offset of the center grating.
            phase_surround (float or Parameter): The cosine phase-offset of the surround grating.
            grey_level (float or Parameter): The mean luminance.
            pixel_boundaries (list or None): Range of values the monitor can display [lower value, upper value]. Default
                is [-1,1].

        Returns: Pixel intensities of the desired Center-Surround stimulus as numpy.ndarray.
        """

        super().__init__()

        self._params_dict = dict(
            location=location,
            size_total=size_total,
            size_center=size_center,
            size_surround=size_surround,
            contrast_center=contrast_center,
            contrast_surround=contrast_surround,
            orientation_center=orientation_center,
            orientation_surround=orientation_surround,
            spatial_frequency_center=spatial_frequency_center,
            phase_center=phase_center,
            grey_level=grey_level,
            spatial_frequency_surround=spatial_frequency_surround,
            phase_surround=phase_surround
        )

        location = self.parse_parameter(location)
        size_total = self.parse_parameter(size_total)
        size_center = self.parse_parameter(size_center)
        size_surround = self.parse_parameter(size_surround)
        contrast_center = self.parse_parameter(contrast_center)
        contrast_surround = self.parse_parameter(contrast_surround)
        orientation_center = self.parse_parameter(orientation_center)
        orientation_surround = self.parse_parameter(orientation_surround)
        spatial_frequency_center = self.parse_parameter(spatial_frequency_center)
        phase_center = self.parse_parameter(phase_center)
        grey_level = self.parse_parameter(grey_level)
        spatial_frequency_surround = self.parse_parameter(spatial_frequency_surround)
        phase_surround = self.parse_parameter(phase_surround)

        self.location = nn.Parameter(torch.Tensor(location))
        self.size_total = nn.Parameter(torch.Tensor([size_total]))
        self.size_center = nn.Parameter(torch.Tensor([size_center]))
        self.size_surround = nn.Parameter(torch.Tensor([size_surround]))
        self.contrast_center = nn.Parameter(torch.Tensor([contrast_center]))
        self.contrast_surround = nn.Parameter(torch.Tensor([contrast_surround]))
        self.orientation_center = nn.Parameter(torch.Tensor([orientation_center]))
        self.orientation_surround = nn.Parameter(torch.Tensor([orientation_surround]))
        self.spatial_frequency_center = nn.Parameter(torch.Tensor([spatial_frequency_center]))
        self.phase_center = nn.Parameter(torch.Tensor([phase_center]))
        self.grey_level = nn.Parameter(torch.Tensor([grey_level]))
        self.spatial_frequency_surround = nn.Parameter(torch.Tensor([spatial_frequency_surround]))
        self.phase_surround = nn.Parameter(torch.Tensor([phase_surround]))

        self.canvas_size = canvas_size
        self.pixel_boundaries = [-1, 1] if pixel_boundaries is None else pixel_boundaries

    def get_config(self):
        return dict(
            location=self.location.detach().numpy().tolist(),
            size_total=self.size_total.item(),
            size_center=self.size_center.item(),
            size_surround=self.size_surround.item(),
            contrast_center=self.contrast_center.item(),
            contrast_surround=self.contrast_surround.item(),
            orientation_center=self.orientation_center.item(),
            orientation_surround=self.orientation_surround.item(),
            spatial_frequency_center=self.spatial_frequency_center.item(),
            phase_center=self.phase_center.item(),
            grey_level=self.grey_level.item(),
            spatial_frequency_surround=self.spatial_frequency_surround.item(),
            phase_surround=self.phase_surround.item()
        )

    def forward(self):
        return self.center_surround(
            self.canvas_size,
            self.location,
            self.size_total,
            self.size_center,
            self.size_surround,
            self.contrast_center,
            self.contrast_surround,
            self.orientation_center,
            self.orientation_surround,
            self.spatial_frequency_center,
            self.spatial_frequency_surround,
            self.phase_center,
            self.phase_surround,
            self.grey_level,
            self.pixel_boundaries
        )

    @staticmethod
    def center_surround(canvas_size, location, size_total, size_center, size_surround, contrast_center,
                        contrast_surround,
                        orientation_center, orientation_surround, spatial_frequency_center, spatial_frequency_surround,
                        phase_center, phase_surround, grey_level, pixel_boundaries):
        x, y = torch.meshgrid(torch.arange(canvas_size[0]) - location[0],
                              torch.arange(canvas_size[1]) - location[1])

        R_center = torch.tensor([[torch.cos(orientation_center), -torch.sin(orientation_center)],
                                 [torch.sin(orientation_center), torch.cos(orientation_center)]])

        R_surround = torch.tensor([[torch.cos(orientation_surround), -torch.sin(orientation_surround)],
                                   [torch.sin(orientation_surround), torch.cos(orientation_surround)]])

        coords = torch.stack([x.flatten(), y.flatten()])
        x_center, y_center = R_center.matmul(coords).reshape((2,) + x.shape)
        x_surround, y_surround = R_surround.matmul(coords).reshape((2,) + x.shape)

        norm_xy_center = torch.sqrt(x_center ** 2 + y_center ** 2)
        norm_xy_surround = torch.sqrt(x_surround ** 2 + y_surround ** 2)

        envelope_center = (norm_xy_center <= size_center * size_total)
        envelope_surround = (norm_xy_surround > size_surround * size_total) * (norm_xy_surround <= size_total)

        grating_center = torch.cos(spatial_frequency_center * x_center * (2 * torch.pi) + phase_center)
        grating_surround = torch.cos(spatial_frequency_surround * x_surround * (2 * torch.pi) + phase_surround)

        # add contrast
        amplitude_center = contrast_center * min(abs(pixel_boundaries[0] - grey_level),
                                                 abs(pixel_boundaries[1] - grey_level))
        amplitude_surround = contrast_surround * min(abs(pixel_boundaries[0] - grey_level),
                                                     abs(pixel_boundaries[1] - grey_level))

        grating_center_contrast = amplitude_center * grating_center
        grating_surround_contrast = amplitude_surround * grating_surround

        center_surround = envelope_center * grating_center_contrast + envelope_surround * grating_surround_contrast

        return center_surround.view((1, 1, *canvas_size)).cuda()
