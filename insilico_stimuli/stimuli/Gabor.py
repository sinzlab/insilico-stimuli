import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

from .Stimulus import Stimulus


class Gabor(nn.Module, Stimulus):
    """
        A class to generate a Gabor stimulus.
    """
    def __init__(self, canvas_size, theta, sigma, Lambda, psi, gamma, center):
        """
        Args:
            theta (float or Parameter): Orientation of the sinusoid (in radian).
            sigma (float or Parameter): std deviation of the Gaussian.
            Lambda (float or Parameter): Sinusoid wavelength (1/frequency).
            psi (float or Parameter): Phase of the sinusoid.
            gamma (float or Parameter): The ratio between sigma in x-dim over sigma in y-dim (acts
                like an aspect ratio of the Gaussian).
            center (tuple of integers or Parameter): The position of the filter.
            canvas_size (tuple of integers): Image height and width.
        Returns:
            2D torch.tensor: A gabor filter.
        """

        super().__init__()

        self._params_dict = dict(
            theta=theta,
            sigma=sigma,
            Lambda=Lambda,
            psi=psi,
            gamma=gamma,
            center=center
        )

        theta = self.parse_parameter(theta)
        sigma = self.parse_parameter(sigma)
        Lambda = self.parse_parameter(Lambda)
        psi = self.parse_parameter(psi)
        gamma = self.parse_parameter(gamma)
        center = self.parse_parameter(center)

        self.theta = nn.Parameter(torch.Tensor([theta]))
        self.sigma = nn.Parameter(torch.Tensor([sigma]))
        self.Lambda = nn.Parameter(torch.Tensor([Lambda]))
        self.psi = nn.Parameter(torch.Tensor([psi]))
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.center = nn.Parameter(torch.Tensor(center))

        self.canvas_size = canvas_size

    def get_config(self):
        return dict(
            theta=self.theta.item(),
            sigma=self.sigma.item(),
            Lambda=self.Lambda.item(),
            psi=self.psi.item(),
            gamma=self.gamma.item(),
            center=self.center.detach().numpy().tolist(),
        )

    def forward(self):
        return self.gabor(
            self.canvas_size,
            self.theta,
            self.sigma,
            self.Lambda,
            self.psi,
            self.gamma,
            self.center
        )

    @staticmethod
    def gabor(canvas_size, theta, sigma, Lambda, psi, gamma, center):
        # clip values in reasonable range
        theta.data.clamp_(-torch.pi, torch.pi)
        sigma.data.clamp_(3., min(canvas_size) / 2)  # min(self.canvas_size)/7, min(self.canvas_size)/5) #2)

        sigma_x = sigma
        sigma_y = sigma / gamma

        ymax, xmax = canvas_size
        xmax, ymax = (xmax - 1) / 2, (ymax - 1) / 2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax + 1), torch.arange(xmin, xmax + 1))

        # Rotation
        x_theta = (x - center[0]) * torch.cos(theta) + (y - center[1]) * torch.sin(theta)
        y_theta = -(x - center[0]) * torch.sin(theta) + (y - center[1]) * torch.cos(theta)

        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(
            2 * torch.pi / Lambda * x_theta + psi)

        return gb.view((1, 1, *canvas_size)).cuda()

    def apply_changes(self):
        self.sigma.requires_grad_(True)
