import torch
from torch import nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

from .Gabor import Gabor

class PlaidsGabor(nn.Module):
    """
        A class to generate a Gabor Plaids stimulus.
    """
    def __init__(self, canvas_size, theta, sigma, Lambda, psi, gamma, angle, beta, center):
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
            angle (float or Parameter): angle between the two gabors.
            beta (float or Parameter): ratio between the two gabors
        Returns:
            2D torch.tensor: A gabor filter.
        """

        super().__init__()
        self.theta = nn.Parameter(torch.Tensor([theta]))
        self.sigma = nn.Parameter(torch.Tensor([sigma]))
        self.Lambda = nn.Parameter(torch.Tensor([Lambda]))
        self.psi = nn.Parameter(torch.Tensor([psi]))
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.angle = nn.Parameter(torch.Tensor([angle]))
        self.beta = nn.Parameter(torch.Tensor([beta]))
        self.center = nn.Parameter(torch.Tensor(center))

        self.canvas_size = canvas_size

    def get_config(self):
        return dict(
            theta=self.theta.item(),
            sigma=self.sigma.item(),
            Lambda=self.Lambda.item(),
            psi=self.psi.item(),
            gamma=self.gamma.item(),
            angle=self.angle.item(),
            beta=self.beta.item(),
            center=self.center.detach().numpy().tolist()
        )

    def forward(self):
        a = Gabor(self.canvas_size,
            self.theta,
            self.sigma,
            self.Lambda,
            self.psi,
            self.gamma,
            self.center)

        b = Gabor(self.canvas_size,
            self.theta + self.angle,
            self.sigma,
            self.Lambda,
            self.psi,
            self.gamma,
            self.center)

        return a() + self.beta * b()


    def apply_changes(self):
        self.sigma.requires_grad_(True)
