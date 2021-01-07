from nnidentify.fitgabor.fitgabor import GaborGenerator, trainer_fn

from torch import nn
import torch


def gradient_descent(StimulusSet, model, fixed_std=1., lr=5e-3, epochs=20000, StimulusGenerator=GaborGenerator, trainer_fn=trainer_fn):
    init_params = StimulusSet.params_dict_from_idx(0)
    init_params['canvas_size'] = StimulusSet.canvas_size

    gabor_gen = StimulusGenerator(init_params['canvas_size'][::-1], target_std=fixed_std)

    test_img = gabor_gen()
    activations = model(test_img).detach().numpy().squeeze()

    params = []
    max_activations = []

    for unit in range(len(activations)):
        gabor_gen.theta = nn.Parameter(torch.Tensor([init_params['orientation']]))
        gabor_gen.center = nn.Parameter(torch.Tensor(init_params['location']))
        gabor_gen.Lambda = nn.Parameter(torch.Tensor([1 / init_params['spatial_frequency']]))
        gabor_gen.psi = nn.Parameter(torch.Tensor([init_params['phase']]))
        gabor_gen.gamma = nn.Parameter(torch.Tensor([init_params['gamma']]), requires_grad=False)

        gabor_gen, _ = trainer_fn(gabor_gen, lambda x: model(x)[:, unit], epochs=epochs, lr=lr, fixed_std=fixed_std)

        max_img = gabor_gen()
        max_activation = model(max_img).detach().numpy().squeeze()[unit]

        stimuli_params = init_params.copy()

        stimuli_params['orientation'] = gabor_gen.theta.detach().cpu().numpy()
        stimuli_params['location'] = gabor_gen.center.detach().cpu().numpy()
        stimuli_params['spatial_frequency'] = 1 / gabor_gen.Lambda.detach().cpu().numpy()
        stimuli_params['phase'] = gabor_gen.psi.detach().cpu().numpy()
        stimuli_params['gamma'] = gabor_gen.gamma.detach().cpu().numpy()

        params.append(stimuli_params)
        max_activations.append(max_activation)

    return params, max_activations