# insilico-stimuli
A toolset for generating image stimuli for insilico experiments. 

There are 6 different types of stimuli: bars, gabors, plaids from gabors, difference of gaussians, 
center surrounds, plaids from circular gratings. 

- search methods (become active if inputs are objects from parameters.py) implemented for bar, gabor,
  dog, centersurround. Not implemented for the 2 plaids classes.
- generation methods (implemented in Base Class)

- when adding a class, always add the methods params(), _parameter_converter() and stimulus() for 
  generation and always add _param_dict_for_search() for search methods.

- the parameters.py module has some untested elements in it, esp. the continuous sampling 
- did not account for potential aliasing effects

```python
import numpy as np
from insilico_stimuli.stimuli_parameters import * 

# Bar stimulus
canvas_size  = [41, 41]
locations    = [[20.2, 20.2]]
lengths      = [11.0]
widths       = [5.0]
contrasts    = [-1] 
orientations = [0.0, np.pi/4, np.pi/2] 
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
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')


# Gabor stimulus
canvas_size         = [41, 41]
sizes               = [15.0]
spatial_frequencies = [1/20, 1/5]
contrasts           = [0.5, 1.0]
grey_levels         = [0.0]
eccentricities      = [0.0, 0.9]
locations           = [[25.0, 30.0]]
orientations        = [val * (np.pi) / 4 for val in range(0, 4)]  
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
    plt.subplot(8, 8, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')


# Plaids (based on Gabors)
canvas_size         = [41, 41]
locations           = [[20.0, 20.0]]
sizes               = [20.0]
spatial_frequencies = [3/10]
orientations        = list(np.arange(0, np.pi, np.pi/8))
phases              = [0.0]
contrasts_preferred = [0.5, 1.0]
contrasts_overlap   = [0.75]
grey_levels         = [0.0]
angles              = list(np.arange(0, np.pi, np.pi/8))

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
    plt.subplot(5, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')


# Difference of Gaussians
canvas_size              = [41, 41]
locations                = [[25.0, 18.0]]
sizes                    = [25.0]
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


# Center Surround
canvas_size                = [41, 41]
locations                  = [[20.0, 20.0]]  # center position
sizes_total                = [17.0]          # total size (center + surround)
sizes_center               = [0.7]           # portion of radius used for center circle
sizes_surround             = [0.7, 0.8]      # defines the starting portion of radius for surround
contrasts_center           = [0.75, 1.0]     # try 2 center contrasts
contrasts_surround         = [0.75]          # surround contrast
orientations_center        = [0.0, np.pi/2]  # variable orientation
orientations_surround      = [0.0, np.pi/4]  # center only
spatial_frequencies_center = [0.2]           # fixed spatial frequency
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
    plt.subplot(8, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')


# Center surround (only circular grating)
canvas_size                = [41, 41]
locations                  = [[20.0, 20.0]]  # fixed position
sizes_total                = [17.0]          # variable size within range [5.0, 13.0]
sizes_center               = [1.0]           # only center
sizes_surround             = [1.1]           # doesn't matter, only center (has to be >1)
contrasts_center           = [0.75, 1.0]     # try 2 contrasts
contrasts_surround         = [0.0]           # doesn't matter, only center
orientations_center        = [0.0, np.pi]    # 2 orientations
orientations_surround      = [0.0]           # doesn't matter, center only
spatial_frequencies_center = [0.2]           # fixed spatial frequency
phases_center              = [0.0, 2*np.pi]  # 2 center phases 
grey_levels                = [0.0]           # fixed grey level

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
    plt.subplot(1, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')


# Plaids (based on circular gratings)
canvas_size          = [41, 41]
locations            = [[20.0, 20.0]]
sizes_total          = [12.0]
spatial_frequencies  = [0.1]
orientations         = [np.pi/4]
phases               = [0.0]
contrasts_preferred  = [0.5, 0.75, 1.0]
contrasts_overlap    = [0.75]
grey_levels          = [0.0]
angles               = list(np.arange(0, np.pi/8, np.pi/4, np.pi/2))  # default, if not specified is pi/2

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
    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
    plt.axis('off')
```