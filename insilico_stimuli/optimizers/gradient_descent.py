from nnidentify.fitgabor.fitgabor import GaborGenerator, trainer_fn

from torch import nn
import torch


def gradient_descent(StimulusSet, model, fixed_std=1., lr=5e-3, epochs=20000):
    gabor_gen = GaborGenerator(StimulusSet.canvas_size[::-1], target_std=fixed_std)

    test_img = gabor_gen()
    activations = model(test_img.cuda()).detach().cpu().numpy().squeeze()

    max_params = []
    max_activations = []

    for unit in range(len(activations)):
        gabor_gen, _ = trainer_fn(gabor_gen, lambda x: model(x)[:, unit], epochs=epochs, lr=lr, fixed_std=fixed_std)

        max_img = gabor_gen()
        max_activation = model(max_img.cuda()).detach().cpu().numpy().squeeze()[unit]

        stimuli_params = {
            'canvas_size': StimulusSet.canvas_size,
            'theta': gabor_gen.theta.detach().cpu().numpy(),
            'sigma': gabor_gen.sigma.detach().cpu().numpy(),
            'Lambda': gabor_gen.Lambda.detach().cpu().numpy(),
            'psi': gabor_gen.psi.detach().cpu().numpy(),
            'gamma': gabor_gen.gamma.detach().cpu().numpy(),
            'center': gabor_gen.center.detach().cpu().numpy(),
            'image': gabor_gen().detach().cpu().numpy()
        }

        max_params.append(stimuli_params)
        max_activations.append(max_activation)

    return max_params, max_activations