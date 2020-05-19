import numpy as np
from numpy import pi


class StimuliSet:
    def __init__(self):
        pass

    def params(self):
        raise NotImplementedError

    def num_params(self):
        return [len(p[0]) for p in self.params()]

    def stimulus(self, *args, **kwargs):
        raise NotImplementedError

    def params_from_idx(self, idx):
        num_params = self.num_params()
        c = np.unravel_index(idx, num_params)
        params = [p[0][c[i]] for i, p in enumerate(self.params())]
        return params

    def params_dict_from_idx(self, idx):
        params = self.params_from_idx(idx)
        return {p[1]: params[i] for i, p in enumerate(self.params())}

    def stimulus_from_idx(self, idx):
        return self.stimulus(**self.params_dict_from_idx(idx))

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params())
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.stimulus_from_idx(i)
                          for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params())
        return np.array([self.stimulus_from_idx(i) for i in range(num_stims)])


class GaborSet(StimuliSet):
    def __init__(self,
                 canvas_size,  # width x height
                 center_range, # [x_start, x_end, y_start, y_end]
                 sizes, # +/- 2 SD of envelope
                 spatial_frequencies,  # cycles / envelop SD, i.e. depends on size
                 contrasts,
                 orientations,
                 phases,
                 relative_sf=True):   # scale SF by size (True) or use absolute units (False)

        self.canvas_size = canvas_size
        self.cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(self.cr[0], self.cr[1])
                    for y in range(self.cr[2], self.cr[3])])
        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations
        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases

        self.relative_sf = relative_sf

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.contrasts, 'contrast'),
            (self.orientations, 'orientation'),
            (self.phases, 'phase')
        ]

    def stimulus(self, location, size, spatial_frequency, contrast, orientation, phase, **kwargs):
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation),  np.cos(orientation)]])
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2, ) + x.shape)
        envelope = 0.5 * contrast * np.exp(-(x ** 2 + y ** 2) / (2 * (size/4)**2))

        grating = np.cos(spatial_frequency * x * (2*pi) + phase)
        return envelope * grating


class PlaidsSet(GaborSet):
    def __init__(self,
                 canvas_size,
                 center,
                 size,
                 spatial_frequency,
                 orientation,
                 phase,
                 contrasts_preferred,
                 contrasts_orthogonal,
                 relative_sf=True):

        self.canvas_size = canvas_size
        self.center = center
        self.size = size
        self.spatial_frequency = spatial_frequency
        self.orientation = orientation
        self.phase = phase
        self.contrasts_preferred = contrasts_preferred
        self.contrasts_orthogonal = contrasts_orthogonal
        self.relative_sf = relative_sf

    def params(self):
        return [
            (self.contrasts_preferred, 'contrast_preferred'),
            (self.contrasts_orthogonal, 'contrast_orthogonal')
        ]

    def stimulus(self, contrast_preferred, contrast_orthogonal, **kwargs):
        gabor_preferred = super().stimulus(
            location=self.center,
            size=self.size,
            spatial_frequency=self.spatial_frequency,
            orientation=self.orientation,
            phase=self.phase,
            contrast=contrast_preferred
        )

        gabor_orthogonal = super().stimulus(
            location=self.center,
            size=self.size,
            spatial_frequency=self.spatial_frequency,
            orientation=self.orientation + np.pi/2,
            phase=self.phase,
            contrast=contrast_orthogonal
        )

        return gabor_preferred + gabor_orthogonal


class DiffOfGaussians(StimuliSet):
    def __init__(self,
                 canvas_size,  # width x height
                 center_range, # [x_start, x_end, y_start, y_end]
                 sizes,
                 sizes_scale_surround,
                 contrasts,
                 contrasts_scale_surround):

        self.canvas_size = canvas_size
        self.cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(self.cr[0], self.cr[1])
                    for y in range(self.cr[2], self.cr[3])])
        self.sizes = sizes
        self.sizes_scale_surround = sizes_scale_surround
        self.contrasts = contrasts
        self.contrasts_scale_surround = contrasts_scale_surround

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes, 'size'),
            (self.sizes_scale_surround, 'size_scale_surround'),
            (self.contrasts, 'contrast'),
            (self.contrasts_scale_surround, 'contrast_scale_surround')
        ]

    def gaussian_density(self, coords, mean, scale):
        mean = np.reshape(mean, [1, -1])
        r2 = np.sum(np.square(coords - mean), axis=1)
        return np.exp(-r2 / (2 * scale**2))

    def stimulus(self, location, size, size_scale_surround, contrast, contrast_scale_surround, eps=1e-6, **kwargs):
        x, y = np.meshgrid(np.arange(self.canvas_size[0]),
                           np.arange(self.canvas_size[1]))
        coords = np.stack([x.flatten(), y.flatten()], axis=-1).reshape(-1, 2)

        center = self.gaussian_density(
            coords,
            mean=location,
            scale=size
        ).reshape(self.canvas_size[::-1])

        surround = self.gaussian_density(
            coords,
            mean=location,
            scale=(size_scale_surround * size)
        ).reshape(self.canvas_size[::-1])

        center_surround = center - contrast_scale_surround * surround
        center_surround = contrast * center_surround
        return center_surround


class CenterSurround(StimuliSet):
    def __init__(self,
                 canvas_size,    # width x height
                 center_range,   # [x_start, x_end, y_start, y_end]
                 sizes_total,
                 sizes_center,   # fraction of total size
                 sizes_surround, # fraction of total size
                 contrasts_center,
                 contrasts_surround,
                 orientations_center,
                 orientations_surround,
                 spatial_frequencies,
                 phases,
                 relative_sf=True):

        self.canvas_size = canvas_size
        self.cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(self.cr[0], self.cr[1])
                    for y in range(self.cr[2], self.cr[3])])
        self.sizes_total = sizes_total
        self.sizes_center = sizes_center
        self.sizes_surround = sizes_surround
        self.contrasts_center = contrasts_center
        self.contrasts_surround = contrasts_surround

        if type(orientations_center) is not list:
            self.orientations_center = np.arange(orientations_center) * pi / orientations_center
        else:
            self.orientations_center = orientations_center

        if type(orientations_surround) is not list:
            self.orientations_surround = np.arange(orientations_surround) * pi / orientations_surround
        else:
            self.orientations_surround = orientations_surround

        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases

        self.spatial_frequencies = spatial_frequencies
        self.relative_sf = relative_sf

    def params(self):
        return [
            (self.locations, 'location'),
            (self.sizes_total, 'size_total'),
            (self.sizes_center, 'size_center'),
            (self.sizes_surround, 'size_surround'),
            (self.contrasts_center, 'contrast_center'),
            (self.contrasts_surround, 'contrast_surround'),
            (self.orientations_center, 'orientation_center'),
            (self.orientations_surround, 'orientation_surround'),
            (self.spatial_frequencies, 'spatial_frequency'),
            (self.phases, 'phase')
        ]

    def stimulus(
        self,
        location,
        size_total,
        size_center,
        size_surround,
        contrast_center,
        contrast_surround,
        orientation_center,
        orientation_surround,
        spatial_frequency,
        phase):

        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])

        R_center = np.array([[np.cos(orientation_center), -np.sin(orientation_center)],
                             [np.sin(orientation_center),  np.cos(orientation_center)]])

        R_surround = np.array([[np.cos(orientation_surround), -np.sin(orientation_surround)],
                               [np.sin(orientation_surround),  np.cos(orientation_surround)]])

        coords = np.stack([x.flatten(), y.flatten()])
        x_center, y_center = R_center.dot(coords).reshape((2, ) + x.shape)
        x_surround, y_surround = R_surround.dot(coords).reshape((2, ) + x.shape)

        norm_xy_center = np.sqrt(x_center ** 2 + y_center ** 2)
        norm_xy_surround = np.sqrt(x_surround ** 2 + y_surround ** 2)

        envelope_center = contrast_center * (norm_xy_center <= size_center * size_total)
        envelope_surround = contrast_surround * \
                            (norm_xy_surround > size_surround * size_total) * \
                            (norm_xy_surround <= size_total)

        grating_center = np.cos(spatial_frequency * x_center * (2*pi) + phase)
        grating_surround = np.cos(spatial_frequency * x_surround * (2*pi) + phase)
        return  envelope_center * grating_center + envelope_surround * grating_surround
