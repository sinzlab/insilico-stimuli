import numpy as np


class Parameter:
    """ A class that can sample from a parameter and return its range. """
    def __init__(self, interval, n=1):
        """
        Args:
            n (int or None): number of samples, default=1.
            interval (list): indicates the parameter range.
        """
        self.low, self.high = interval
        self.n = n

    def sample(self, cdf_inv=None):
        """
        Args:
            cdf_inv (func or None): Pseudo-inverse of the cumulative density function defined on self.interval.
                This function needs to be able to get just one numpy.array as an input.

        Returns:
            numpy.array: returns n random samples from this parameter with the inverse transformation method.
        """
        if cdf_inv is None:
            return np.random.uniform(low=self.low, high=self.high, size=self.n)
        else:
            u = np.random.uniform(low=0, high=1, size=self.n)
            return cdf_inv(u)

    def range(self):
        """
        Returns:
            list: the range of the parameter.
        """
        return [self.low, self.high]


class FiniteParameter(Parameter):
    """ Subclass of Parameter which can hold specific, unsampled parameter values. """
    def __init__(self, val):
        """
        Args:
            val (list or int or float): defines the parameter values.
        """
        self.values = val
        if isinstance(val, list):
            if len(val) > 1:
                self.low = min(val)
                self.high = max(val)

    @property
    def values(self):
        """
        Returns:
            list or int or float: set of parameter values
        """
        return self._values

    @values.setter
    def values(self, val):
        if isinstance(val, (list, int, float)):
            self._values = val
        else:
            raise TypeError('val must be either of type list, float or int.')

    def sample(self):
        raise NotImplementedError('sample method not supported for objects of type FiniteParameter.')

class FiniteSelection(FiniteParameter):
    """ Subclass of FiniteParameter which can draw random samples from a given finite set of parameter values. """
    def __init__(self, values, pmf=None, n=1):
        """
        Args:
            values (list): contains the desired parameter values.
            pmf (list or None): probability mass function, default is uniform distribution.
            n (int or None): number of samples, default=1
        """
        self._values = values
        if pmf is None:
            n_values = len(self._values)
            self.pmf = list(np.ones(n_values) / n_values)
        else:
            self.pmf = pmf
        self.n = n

    @property
    def values(self):
        return self._values

    def sample(self):
        """
        Returns:
            list: n samples of values drawn from pmf.
        """
        idx = list(np.random.choice(np.arange(len(self._values)), size=self.n, p=self.pmf))
        res = []
        for i in idx:
            res.append(self._values[i])
        return res

    @property
    def range(self):
        """

        Returns:
            list: parameter range.
        """
        return [min(self._values), max(self._values)]


class UniformRange(Parameter):
    """ Subclass of Parameter which can sample the parameter values from an infinite set, defined by the parameter
    boundaries. """
    def __init__(self, interval, n=1, cdf_inv=None):
        """
        Args:
            interval (list of float): parameter range. Can be of the format [[x1_start, x1_end], [x2_start, x2_end]]
            n (int or None): number of samples, default=1.
            cdf_inv (func or None): Pseudo-inverse of the cumulative density function defined on self.interval.
                 This function needs to be able to get just one numpy.array as an input.
        """
        super().__init__(interval, n)
        self.interval = interval
        self.cdf_inv = cdf_inv

    def sample(self, n=None):
        """
        Args:
            n (int or None): number of samples.

        Returns:
             list: returns n random samples from this parameter.
         """
        if n is not None:
            self.n = n

        if isinstance(self.low, list) and isinstance(self.high, list):
            if self.cdf_inv is None:
                s = np.zeros((self.n, 2))  # samples are stored row-wise
                for param_idx in range(0, 2):
                    s[:, param_idx] = np.random.uniform(low=self.interval[param_idx][0],
                                                        high=self.interval[param_idx][1],
                                                        size=self.n)
                return [list(s[sample_idx, :]) for sample_idx in range(0, self.n)]
            else:
                raise SyntaxError("If the interval is a list of lists, a cdf input is not supported.")
        elif isinstance(self.low, (float, int)) and isinstance(self.high, (float, int)):
            if self.cdf_inv is None:
                return list(np.random.uniform(low=self.low, high=self.high, size=self.n))
            else:
                u = np.random.uniform(low=0, high=1, size=self.n)
                return self.cdf_inv(u)

    @property
    def range(self):
        """
        Returns:
            list: the range of the parameter.
        """
        return super().range()
