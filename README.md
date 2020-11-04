# insilico-stimuli
A toolset for generating image stimuli for insilico experiments. 

There are 6 different types of stimuli: bars, gabors, plaids from gabors, difference of gaussians, 
center surrounds, plaids from circular gratings. 

Except from the two plaid classes, all stimulus classes have implemented search methods. One search method is based on 
an optimizer by ax-platform, which looks for optimal parameters via Bayesian search. You can specify the number of 
optimization loops, it should perform. The optimizer can deal with hybrid parameter combinations, 
meaning one argument (e.g. `spatial_frequencies`) can be a continuous parameter within a given range while
another argument (e.g. `contrasts`) can be a fixed parameter. This search method becomes active if inputs are objects
from `parameters.py`. The other search method is a bruteforce method. It only works when no input parameter is of type
`UniformRange`. It will try out every parameter combination from the inputs and find that parameter combination, 
which yields highest model activation. 
The general recommendation is to use the bruteforce search, if the search is over a discrete range of parameter values,
whereas the Bayesian search should be used if at least one parameter should be optimized over a continuous range of 
values. More precisely, the bruteforce search will throw an error when trying to use it to solve continuous searching 
tasks. For "finite searches", the bruteforce search is more accurate than the Bayesian search.


Apart from the search methods, the major feature of this toolbox is to generate stimuli of all kinds. There
are examples for every class below.

Files in this folder:

++++ insilico_stimuli ++++
- `parameters.py` defines the object types that the stimulus arguments can take on.
- `stimuli.py` defines the stimulus classes with its generation and search methods

++++ notebooks ++++
- `bar_endstopping_experiment.ipynb` demonstrate the phenomenon of end-stopping in V1 neurons with one arbitrary model 
unit.
- `comparison_search_methods.ipynb` is a notebook assessing whether the outcome of the two implemented search methods 
yield identical results. Also, it compares the MEIs from 28 model neurons with the optimal Gabors from the Bayesian 
search and the bruteforce search.
- `parameters_examples.ipynb` is a demo notebook for the `parameters.py` module.
- `plaids_tuning_experiment.ipynb` is a notebook which conducts orientation tuning and contrast tuning experiments and 
creates plaid tuning curves in a toy experiment way.
- `size_tuning_experiment.ipynb` is an example notebook how this toolbox could be used when conducting a size tuning 
experiment 
- `size_tuning_multisessions.ipynb` is a notebook which adapted the bruteforce search method in a way that it can handle
several data keys at once. It can be used to further develop the toolbox.
- `stimuli_examples.ipynb` features examples of all stimulus class for all kinds of class configurations. It can be seen 
as a more detailed version of the examples (see below) given in this readme file. 


_Some additional notes_:
- when adding a class, always add the methods `params()`, `_parameter_converter()` and `stimulus()` for 
  generation and always add `_param_dict_for_search()` for search methods.
- the stimuli are not accounted for potential aliasing effects at the stimulus edges
- data type needs to be `float` for all list elements when calling a class with objects from `parameters.py`

# Stimulus Generation Demo

```python
import numpy as np
import matplotlib.pyplot as plt

import insilico_stimuli
from insilico_stimuli.stimuli import BarsSet, GaborSet, PlaidsGaborSet, DiffOfGaussians, CenterSurround, PlaidsGratingSet
from insilico_stimuli.parameters import *

# Bar stimulus
canvas_size  = [41, 41]
locations    = [[20.2, 20.2]]
lengths      = [9.0, 12.0]
widths       = [5.0, 7.0]
contrasts    = [-1] 
orientations = [0.0, np.pi/4] 
grey_levels  = [0.0]

# Instantiate the bar stimulus class
bar_set = BarsSet(canvas_size=canvas_size,
                  locations=locations,
                  lengths=lengths,
                  widths=widths,
                  contrasts=contrasts,
                  orientations=orientations,
                  grey_levels=grey_levels)

# plot the generated stimuli
plt.figure(figsize=(10, 5))
for i, img in enumerate(bar_set.images()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_bars](https://user-images.githubusercontent.com/52453661/97459460-179f9a80-193c-11eb-824f-f966cae3e25e.JPG)


```python
# Gabor stimulus
canvas_size         = [41, 41]
sizes               = [15.0]
spatial_frequencies = [1/20, 1/5]
contrasts           = [0.5, 1.0]
grey_levels         = [0.0]
eccentricities      = [0.0, 0.9]
locations           = [[25.0, 24.0]]
orientations        = [val * (np.pi) / 4 for val in range(0, 2)]  
phases              = [np.pi/2, 3*np.pi/2]

# Instantiate the Gabor class
gabor_set = GaborSet(canvas_size=canvas_size,
                     sizes=sizes,
                     spatial_frequencies=spatial_frequencies,
                     contrasts=contrasts,
                     orientations=orientations,
                     phases=phases, 
                     grey_levels=grey_levels,
                     eccentricities=eccentricities,
                     locations=locations,
                     relative_sf=False)

# plot the generated stimuli
plt.figure(figsize=(10, 5))
for i, img in enumerate(gabor_set.images()):
    plt.subplot(4, 8, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_gabors](https://user-images.githubusercontent.com/52453661/97459912-8d0b6b00-193c-11eb-8275-c769c893f976.JPG)

```python
# Plaids (based on Gabors)
canvas_size         = [41, 41]
locations           = [[20.0, 20.0]]
sizes               = [20.0]
spatial_frequencies = [2/10]
orientations        = list(np.arange(0, np.pi, np.pi/4))
phases              = [0.0]
contrasts_preferred = [0.5, 1.0]
contrasts_overlap   = [0.75]
grey_levels         = [0.0]
angles              = list(np.arange(0, np.pi, np.pi/4))

# instantiate plaids class
plaids_set = PlaidsGaborSet(canvas_size=canvas_size, 
                            locations=locations,
                            sizes=sizes,
                            spatial_frequencies=spatial_frequencies,
                            orientations=orientations,
                            phases=phases,
                            contrasts_preferred=contrasts_preferred,
                            contrasts_overlap=contrasts_overlap, 
                            grey_levels=grey_levels, 
                            angles=angles)

# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(plaids_set.images()):
    plt.subplot(4, 8, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_plaids](https://user-images.githubusercontent.com/52453661/97459968-9b598700-193c-11eb-8269-22e884c9f638.JPG)

```python
# Difference of Gaussians
canvas_size              = [41, 41]
locations                = [[25.0, 18.0]]
sizes                    = [10.0]
sizes_scale_surround     = [1.01, 2.0]
contrasts                = [-1.0, 1.0]
contrasts_scale_surround = [0.5, 1.0]
grey_levels              = [0.0]

DoG = DiffOfGaussians(canvas_size=canvas_size, 
                      locations=locations,
                      sizes=sizes,
                      sizes_scale_surround=sizes_scale_surround,
                      contrasts=contrasts,
                      contrasts_scale_surround=contrasts_scale_surround, 
                      grey_levels=grey_levels)
                      #pixel_boundaries=None)

# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(DoG.images()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_dog](https://user-images.githubusercontent.com/52453661/97460011-a7dddf80-193c-11eb-8735-5254760494a8.JPG)

```python
# Center Surround
canvas_size                = [41, 41]
locations                  = [[20.0, 20.0]]  # center position
sizes_total                = [17.0]          # total size (center + surround)
sizes_center               = [0.5]           # portion of radius used for center circle
sizes_surround             = [0.5, 0.7]      # defines the starting portion of radius for surround
contrasts_center           = [0.75, 1.0]     # try 2 center contrasts
contrasts_surround         = [0.75]          # surround contrast
orientations_center        = [0.0, np.pi/2]  # variable orientation
orientations_surround      = [0.0, np.pi/4]  # center only
spatial_frequencies_center = [0.1]           # fixed spatial frequency
phases_center              = [0.0, 2*np.pi]  # center phases 
grey_levels                = [0.0]           # fixed grey level
# spatial_frequencies_surround = [0.1, 0.3]  # optional parameter, default: same as spatial_frequencies_center
# phases_surround = [np.pi/4]                # optional parameter, default: same as phases_center

center_surround = CenterSurround(canvas_size=canvas_size, 
                                 locations=locations,
                                 sizes_total=sizes_total,
                                 sizes_center=sizes_center,
                                 sizes_surround=sizes_surround,
                                 contrasts_center=contrasts_center,
                                 contrasts_surround=contrasts_surround,
                                 orientations_center=orientations_center,
                                 orientations_surround=orientations_surround,
                                 spatial_frequencies_center=spatial_frequencies_center,
                                 phases_center=phases_center,
                                 grey_levels=grey_levels)

# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(center_surround.images()):
    plt.subplot(4, 8, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_cs](https://user-images.githubusercontent.com/52453661/97460047-b0ceb100-193c-11eb-8710-4d4b74e667a8.JPG)

```python
# Center surround (only circular grating)
canvas_size                = [41, 41]
locations                  = [[20.0, 20.0]]    # fixed position
sizes_total                = [17.0]            # variable size within range [5.0, 13.0]
sizes_center               = [1.0]             # only center
sizes_surround             = [1.1]             # doesn't matter, only center (has to be >1)
contrasts_center           = [0.75, 1.0]       # try 2 contrasts
contrasts_surround         = [0.0]             # doesn't matter, only center
orientations_center        = [0.0, np.pi/2]    # 2 orientations
orientations_surround      = [0.0]             # doesn't matter, center only
spatial_frequencies_center = [0.1]             # fixed spatial frequency
phases_center              = [0.0, 2*np.pi]    # 2 center phases 
grey_levels                = [0.0]             # fixed grey level

# call center-surround
circular_grating = CenterSurround(canvas_size=canvas_size, 
                                 locations=locations,
                                 sizes_total=sizes_total,
                                 sizes_center=sizes_center,
                                 sizes_surround=sizes_surround,
                                 contrasts_center=contrasts_center,
                                 contrasts_surround=contrasts_surround,
                                 orientations_center=orientations_center,
                                 orientations_surround=orientations_surround,
                                 spatial_frequencies_center=spatial_frequencies_center,
                                 phases_center=phases_center,
                                 grey_levels=grey_levels)
               
# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(circular_grating.images()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_cg](https://user-images.githubusercontent.com/52453661/97460083-bb894600-193c-11eb-8539-a6c4ad28fca6.JPG)

```python
# Plaids (based on circular gratings)
canvas_size          = [41, 41]
locations            = [[20.0, 20.0]]
sizes_total          = [12.0]
spatial_frequencies  = [0.1]
orientations         = [np.pi/4]
phases               = [0.0]
contrasts_preferred  = [0.5, 1.0]
contrasts_overlap    = [0.75]
grey_levels          = [0.0]
angles               = list(np.arange(0, np.pi/2, np.pi/8))  # default, if not specified is pi/2

plaid_grating = PlaidsGratingSet(canvas_size=canvas_size, 
                                 sizes_total=sizes_total,
                                 locations=locations,
                                 contrasts_preferred=contrasts_preferred,
                                 contrasts_overlap=contrasts_overlap, 
                                 spatial_frequencies=spatial_frequencies,
                                 orientations=orientations,
                                 phases=phases,
                                 grey_levels=grey_levels, 
                                 angles=angles,
                                 pixel_boundaries=None)

# plot the generated images
plt.figure(figsize=(10, 5))
for i, img in enumerate(plaid_grating.images()):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```
![readme_plaids_cg](https://user-images.githubusercontent.com/52453661/97460135-c93ecb80-193c-11eb-9313-62f6088978fe.JPG)

# Search Methods
For a search over a finite set of parameter values (here: 547.200 parameter combinations), we can use the _bruteforce search method_:
```python
# loading model
ensemble_hash = 'e7a8d0b961ee37ffda0779bfed95f1cf'
model_key = (TrainedModel * TrainedEnsembleModel.Member() & dict(ensemble_hash=ensemble_hash)).fetch("KEY", limit=1, order_by="score DESC")
model = TrainedModel().load_model(model_key, include_dataloader=False)
model.eval()

# model specifics for optimization
data_key = '3631807112901'
unit_idx = 27

# Define the parameter set (finite)
canvas_size         = [41, 41]
sizes               = FiniteParameter([float(val) for val in range(5, 31)][::5])
spatial_frequencies = FiniteParameter([float(val) for val in np.linspace(1/80, 2/5, 8)])
contrasts           = FiniteParameter([1.0])
orientations        = FiniteParameter([float(val) for val in np.linspace(0.0, np.pi, 10)])
phases              = FiniteParameter([float(val) for val in np.linspace(0.0, 2*np.pi, 10)]) 
grey_levels         = FiniteParameter([0.0])
eccentricities      = FiniteParameter([float(val) for val in np.linspace(0.0, 0.99, 6)])
locations           = FiniteParameter([[float(x), float(y)] for x in range(10, 30) 
                                                            for y in range(10, 30)][::14])

gabor_set = GaborSet(canvas_size=canvas_size,
                     locations=locations,
                     sizes=sizes,
                     spatial_frequencies=spatial_frequencies,
                     contrasts=contrasts,
                     orientations=orientations, 
                     phases=phases, 
                     grey_levels=grey_levels,
                     eccentricities=eccentricities)

# run the optimization by evaluating all stimuli (and print the time it takes)
start_time = time.time()
params, _ , acts = gabor_set.find_optimal_stimulus_bruteforce(model=model, data_key=data_key, batch_size=100, return_activations=True)
print("--- %s seconds ---" % (time.time() - start_time))
```
< print statement >

```python
print("optimal parameters: {}, activation:{}".format(params, acts))
```
< print statement >


This is the resulting _optimal Gabor (bruteforce)_ stimulus:
```python
optGab = gabor_set.get_image_from_params(params)
plt.imshow(optGab, cmap='gray', vmin=-1, vmax=1)
plt.title('Best Gabor (BF), unit_idx:{}\nactivation:{}'.format(unit_idx, acts))
```
< plot >


For comparison, this is what the _Bayesian search_ method finds here:
```python
# run the optimization by Bayesian search with 30 steps (and print the time it takes)
start_time = time.time()
params_Bayes, acts_Bayes = gabor_set.find_optimal_stimulus(model = model, 
                        			                       data_key = data_key,
                                    			           unit_idx = unit_idx,
                                               			   total_trials = 30)
print("--- %s seconds ---" % (time.time() - start_time))
```
![runtime_Bayes](https://user-images.githubusercontent.com/52453661/98012547-86c43580-1df9-11eb-9222-cd6c2ce45944.JPG)

```python
print("optimal parameters:\n{},\n\nactivation:\n{}".format(params_Bayes, acts_Bayes[0]['activation']))
```
![Bayes_print_params](https://user-images.githubusercontent.com/52453661/98012487-71e7a200-1df9-11eb-817b-dc9fc8a231fc.JPG)

This is the resulting _optimal Gabor (Bayes)_ stimulus compared with the _optimal Gabor (BF)_:
```python
fig, axs = plt.subplots(1, 2)

optGab = gabor_set.get_image_from_params(params)
axs[0].imshow(optGab, cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('Best Gabor (BF), unit_idx:{}\nactivation:{}'.format(unit_idx, acts))

optGab_Bayes = gabor_set.get_image_from_params(params_Bayes)
axs[1].imshow(optGab_Bayes, cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Best Gabor (Bayes), unit_idx:{}\nactivation:{}'.format(unit_idx, acts_Bayes))
```
< plot >


A detailed demo notebook can be found in the notebooks folder, [here](https://github.com/sinzlab/insilico-stimuli/blob/Parameter_extension/notebooks/stimuli_examples.ipynb).
