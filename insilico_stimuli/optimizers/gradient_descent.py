import numpy as np

import torch
from torch import optim
from torch import nn

from tqdm import trange


def trainer_fn(image_generator, target_model, epochs=2000, lr=1e-3, fixed_norm=None, fixed_mean=None, fixed_std=None,
               save_rf_every_n_epoch=None):
    """
        Trainer function for optimizing parameters using the Adam optimizer

        Args:
            image_generator (stimulus): Instance of a Stimulus class.
            target_model (Encoder): The evaluated model as an encoder class.
            epochs (int): max number of epochs in training, default: 20000
            lr (int): learning rate, default: 5e-3
            fixed_norm (float or None) (optional): Set the norm of the optimized stimulus to a constant value
            fixed_mean (float or None) (optional): Set the mean of the optimized stimulus to a constant value
            fixed_std (float or None) (optional): Set the std of the optimized stimulus to a constant value
            save_rf_every_n_epoch (int or None) (optional): Saves an image every n epochs

        Returns
            - image_generator (stimulus): Instance of a Stimulus class
            - saved_images (list of np.arrays): list of saved images every n-th epoch. Empty if save_rf_every_n_epoch = None
    """
    target_model.eval()
    image_generator.train()

    optimizer = optim.Adam(image_generator.parameters(), lr=lr)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    old_lr = lr
    lr_change_counter = 0

    pbar = trange(epochs, desc="Loss: {}".format(np.nan), leave=True)
    saved_images = []
    for epoch in pbar:
        optimizer.zero_grad()

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if old_lr != current_lr:
            old_lr = current_lr
            lr_change_counter += 1

        if lr_change_counter > 3:
            break

        # generate image
        images = image_generator()

        if fixed_mean is not None:
            images_mean = images.mean(dim=(2, 3), keepdims=True)
            images = images - images_mean + fixed_mean

        if fixed_std is not None:
            images_std = images.std(dim=(2, 3), keepdims=True)
            images = fixed_std * images / images_std

        if fixed_norm is not None:
            images_norm = torch.norm(images)
            images = fixed_norm * images / images_norm

        loss = -target_model(images)
        loss.backward()
        optimizer.step()

        pbar.set_description("Loss: {:.2f}".format(loss.item()))

        if save_rf_every_n_epoch is not None:
            if (epoch % save_rf_every_n_epoch) == 0:
                saved_images.append(images.cpu().data.numpy())

        lr_scheduler.step(-loss)

    image_generator.eval()
    return image_generator, saved_images

class UnitModel(nn.Module):
    def __init__(self, model, unit):
        super().__init__()
        self.model = model
        self.model.eval()

        self.unit = unit

    def forward(self, x):
        return self.model(x)[:, self.unit]

def gradient_descent_per_unit(stimulus, model, unit, epochs, lr, fixed_norm=None, fixed_mean=None, fixed_std=None):
    """
        Finds optimal parameter combination for a single unit based on the gradient descent method.

        Args:
            stimulus (stimulus): Instance of a Stimulus class.
            model (Encoder): The evaluated model as an encoder class.
            unit (int or None): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            epochs (int): max number of epochs in training
            lr (int): learning rate
            fixed_norm (float or None) (optional): Set the norm of the optimized stimulus to a constant value
            fixed_mean (float or None) (optional): Set the mean of the optimized stimulus to a constant value
            fixed_std (float or None) (optional): Set the std of the optimized stimulus to a constant value

        Returns
            - stimuli_params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
    """

    unit_model = UnitModel(model, unit)
    stimulus, _ = trainer_fn(stimulus, unit_model, epochs=epochs, lr=lr, fixed_norm=fixed_norm, fixed_mean=fixed_mean, fixed_std=fixed_std)

    max_img = stimulus()
    max_activation = model(max_img.cuda()).detach().cpu().numpy().squeeze()[unit]

    stimuli_params = stimulus.get_config()

    return stimuli_params, max_activation


def gradient_descent(stimulus, model, unit=None, seed=None, epochs=20000, lr=5e-3, fixed_norm=None, fixed_mean=None, fixed_std=None, **kwargs):
    """
        Finds optimal parameter combination for all units based on the gradient descent method.

        Args:
            stimulus (stimulus): Instance of a Stimulus class.
            model (Encoder): The evaluated model as an encoder class.
            unit (int or None) (optional): unit index of the desired model neuron. If not specified, return the best
                parameters for all model neurons (advised, because search is done for all units anyway).
            seed (int): random seed for reproducibility
            epochs (int): max number of epochs in training, default: 20000
            lr (int): learning rate, default: 5e-3
            fixed_norm (float or None) (optional): Set the norm of the optimized stimulus to a constant value
            fixed_mean (float or None) (optional): Set the mean of the optimized stimulus to a constant value
            fixed_std (float or None) (optional): Set the std of the optimized stimulus to a constant value

        Returns
            - stimuli_params (list of dict): The optimal parameter settings for each of the different units
            - max_activation (np.array of float): The maximal firing rate for each of the units over all images tested
    """

    torch.manual_seed(seed)

    if unit is None:
        test_img = stimulus()
        activations = model(test_img.cuda()).detach().cpu().numpy().squeeze()

        max_params = []
        max_activations = []

        for _unit in range(len(activations)):
            stimuli_params, max_activation = gradient_descent_per_unit(stimulus, model, _unit, epochs, lr, fixed_norm=fixed_norm, fixed_mean=fixed_mean, fixed_std=fixed_std)

            max_params.append(stimuli_params)
            max_activations.append(max_activation)

        return max_params, max_activations

    else:
        stimuli_params, max_activation = gradient_descent_per_unit(stimulus, model, unit, epochs, lr, fixed_norm=fixed_norm, fixed_mean=fixed_mean, fixed_std=fixed_std)

        return stimuli_params, max_activation