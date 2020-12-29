from fitgabor import GaborGenerator, trainer_fn

def gradient_descent(StimulusSet, model, fixed_std=1., lr=5e-3, epochs=20000):
    init_params = StimulusSet.params_dict_from_idx(0)

    gabor_gen = GaborGenerator(init_params['canvas_size'][::-1], target_std=fixed_std)

    test_img = gabor_gen()
    activations = model(test_img).detach().numpy().squeeze()

    params = []
    max_activations = []

    for unit in len(activations):
        gabor_gen.theta = init_params['orientations']
        gabor_gen.center = init_params['locations']
        gabor_gen.Lambda = 1 / init_params['spatial_frequencies']
        gabor_gen.psi = init_params['phases']
        gabor_gen.gamma = init_params['eccentricities']

        gabor_gen, _ = trainer_fn(gabor_gen, lambda x: model(x)[:, unit], epochs=epochs, lr=lr, fixed_std=fixed_std)

        max_img = gabor_gen()
        max_activation = model(max_img).detach().numpy().squeeze()[unit]

        stimuli_params = init_params.copy()
        stimuli_params['orientations'] = gabor_gen.theta
        stimuli_params['locations'] = gabor_gen.center
        stimuli_params['spatial_frequencies'] = 1 / gabor_gen.Lambda
        stimuli_params['phases'] = gabor_gen.psi
        stimuli_params['eccentricities'] = gabor_gen.gamma

        params.append(stimuli_params)
        max_activations.append(max_activation)

    return params, max_activations