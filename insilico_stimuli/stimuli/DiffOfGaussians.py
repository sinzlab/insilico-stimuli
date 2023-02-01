import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

from .Stimulus import Stimulus


class DiffOfGaussians(nn.Module, Stimulus):
    """
        A class to generate a difference of gaussians stimulus.
    """
    def __init__(self, canvas_size, location, size, size_scale_surround, contrast, contrast_scale_surround,
                 grey_level, pixel_boundaries=None):
        """
        Args:
            location (list of float or Parameter): The center position of the DoG.
            size (float or Parameter): Standard deviation of the center Gaussian.
            size_scale_surround (float or Parameter): Scaling factor defining how much larger the standard deviation of the surround
                Gaussian is relative to the size of the center Gaussian. Must have values larger than 1.
            contrast (float or Parameter): Contrast of the center Gaussian in %. Takes values from -1 to 1.
            contrast_scale_surround (float or Parameter): Contrast of the surround Gaussian relative to the center Gaussian.
            grey_level (float or Parameter): mean luminance.

        Returns: Pixel intensities for desired Difference of Gaussians stimulus as numpy.ndarray.
        """

        super().__init__()

        self._params_dict = dict(
            location=location,
            size=size,
            size_scale_surround=size_scale_surround,
            contrast=contrast,
            contrast_scale_surround=contrast_scale_surround,
            grey_level=grey_level
        )

        location = self.parse_parameter(location)
        size = self.parse_parameter(size)
        size_scale_surround = self.parse_parameter(size_scale_surround)
        contrast = self.parse_parameter(contrast)
        contrast_scale_surround = self.parse_parameter(contrast_scale_surround)
        grey_level = self.parse_parameter(grey_level)

        self.location = nn.Parameter(torch.Tensor(location))
        self.size = nn.Parameter(torch.Tensor([size]))
        self.size_scale_surround = nn.Parameter(torch.Tensor([size_scale_surround]))
        self.contrast = nn.Parameter(torch.Tensor([contrast]))
        self.contrast_scale_surround = nn.Parameter(torch.Tensor([contrast_scale_surround]))
        self.grey_level = nn.Parameter(torch.Tensor([grey_level]))

        self.canvas_size = canvas_size
        self.pixel_boundaries = [-1, 1] if pixel_boundaries is None else pixel_boundaries

    def get_config(self):
        return dict(
            location=self.location.detach().numpy().tolist(),
            size=self.size.item(),
            size_scale_surround=self.size_scale_surround.item(),
            contrast=self.contrast.item(),
            contrast_scale_surround=self.contrast_scale_surround.item(),
            grey_level=self.grey_level.item(),
        )

    def forward(self):
        return self.diff_of_gaussians(
            self.canvas_size,
            self.location,
            self.size,
            self.size_scale_surround,
            self.contrast,
            self.contrast_scale_surround,
            self.grey_level,
            self.pixel_boundaries
        )

    @staticmethod
    def gaussian_density(coords, mean, scale):
        """
        Args:
            coords: The evaluation points with shape (#points, 2) as numpy.ndarray.
            mean (int): The mean/location of the Gaussian.
            scale (int): The standard deviation of the Gaussian.

        Returns: Unnormalized Gaussian density values evaluated at the positions in 'coords' as numpy.ndarray.
        """
        mean = torch.reshape(mean, [1, -1])
        r2 = torch.sum(torch.square(coords - mean), axis=1)
        return torch.exp(-r2 / (2 * scale ** 2))

    def diff_of_gaussians(self, canvas_size, location, size, size_scale_surround, contrast, contrast_scale_surround,
                 grey_level, pixel_boundaries):
        x, y = torch.meshgrid(torch.arange(canvas_size[0]),
                           torch.arange(canvas_size[1]))

        coords = torch.stack([x.flatten(), y.flatten()], axis=-1).reshape(-1, 2)

        center = self.gaussian_density(coords, mean=location, scale=size).reshape(canvas_size[::-1])
        surround = self.gaussian_density(coords, mean=location, scale=(size_scale_surround * size))\
            .reshape(canvas_size[::-1])

        center_surround = center - contrast_scale_surround * surround

        # add contrast
        min_val, max_val = center_surround.min(), center_surround.max()
        amplitude_current = max(torch.abs(min_val), torch.abs(max_val))
        amplitude_required = contrast * min(torch.abs(pixel_boundaries[0] - grey_level), torch.abs(pixel_boundaries[1] - grey_level))
        contrast_scaling = amplitude_required / amplitude_current

        diff_of_gaussians = contrast_scaling * center_surround + grey_level

        return diff_of_gaussians.view((1, 1, *canvas_size)).cuda()

    def apply_changes(self):
        self.sigma.requires_grad_(True)
