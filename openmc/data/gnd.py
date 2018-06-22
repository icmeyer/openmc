from collections import Iterable, Callable
from numbers import Real, Integral

from six import add_metaclass
import numpy as np

import openmc.data
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin
from function import Function1D

class XYs1D(Function1D):
    """A one-dimensional tabulated function.

    This class mirrors the XYs1d type from the GNDS format, which itself
    shares many attributes with the TAB1 type from the ENDF6 format.
    A tabulated function is specified by tabulated (x,y) pairs along with
    interpolation rules that determine the values between tabulated pairs.

    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = XYs1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    interpolation : str
        Interpolation scheme identification 

    Attributes
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    interpolation : str
        Interpolation scheme identification
    n_pairs : int
        Number of tabulated (x,y) pairs

    """

    def __init__(self, x, y, interpolation=None):
        if interpolation is None:
            #linear-linear interpolation by default
            self.interpolation = 'lin-lin'
        else:
            self.interpolation = interpolation

        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def __call__(self, x):
        # Check if input is array or scalar
        if isinstance(x, Iterable):
            iterable = True
            x = np.array(x)
        else:
            iterable = False
            x = np.array([x], dtype=float)

        # Create output array
        y = np.zeros_like(x)

        # Get indices for interpolation
        idx = np.searchsorted(self.x, x, side='right') - 1

        xi = self.x[idx]       # low edge of corresponding bins
        xi1 = self.x[idx + 1]  # high edge of corresponding bins
        yi = self.y[idx]
        yi1 = self.y[idx + 1]

        if self.interpolation == 'flat':
            # Histogram
            y = yi

        elif self.interpolation == 'lin-lin':
            # Linear-linear
            y = yi + (x - xi)/(xi1 - xi)*(yi1 - yi)

        elif self.interpolation == 'lin-log':
            # Linear-log
            y = yi + np.log(x/xi)/np.log(xi1/xi)*(yi1 - yi)

        elif self.interpolation == 'log-lin':
            # Log-linear
            y = yi*np.exp((x - xi)/(xi1 - xi)*np.log(yi1/yi))

        elif self.interpolation == 'log-log':
            # Log-log
            y = (yi*np.exp(np.log(x/xi)/np.log(xi1/xi)
                            *np.log(yi1/yi)))

        #In some cases, x values might be outside the tabulated region due only
        #to precision, so we check if they're close and set them equal if so.
        y[np.isclose(x, self.x[0], atol=1e-14)] = self.y[0]
        y[np.isclose(x, self.x[-1], atol=1e-14)] = self.y[-1]

        return y if iterable else y[0]

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def n_pairs(self):
        return len(self.x)

    @x.setter
    def x(self, x):
        cv.check_type('x values', x, Iterable, Real)
        self._x = x

    @y.setter
    def y(self, y):
        cv.check_type('y values', y, Iterable, Real)
        self._y = y

    @interpolation.setter
    def interpolation(self, interpolation):
        cv.check_type('interpolation', interpolation, Iterable, str)
        self._interpolation = interpolation

    def integral(self):
        """Integral of the tabulated function over its tabulated range.

        Returns
        -------
        numpy.ndarray
            Array of same length as the tabulated data that represents partial
            integrals from the bottom of the range to each tabulated point.

        """

        # Create output array
        partial_sum = np.zeros(len(self.x) - 1)

        i_low = 0
        i_high = len(self.x)

        # Get x values and bounding (x,y) pairs
        x0 = self.x[:-1]
        x1 = self.x[1:]
        y0 = self.y[:-1]
        y1 = self.y[1:]

        if self.interpolation == 'flat':
            # Histogram
            partial_sum[i_low:i_high] = y0*(x1 - x0)

        elif self.interpolation == 'lin-lin':
            # Linear-linear
            m = (y1 - y0)/(x1 - x0)
            partial_sum[i_low:i_high] = (y0 - m*x0)*(x1 - x0) + \
                                        m*(x1**2 - x0**2)/2

        elif self.interpolation == 'lin-log':
            # Linear-log
            logx = np.log(x1/x0)
            m = (y1 - y0)/logx
            partial_sum[i_low:i_high] = y0 + m*(x1*(logx - 1) + x0)

        elif self.interpolation == 'log-lin':
            # Log-linear
            m = np.log(y1/y0)/(x1 - x0)
            partial_sum[i_low:i_high] = y0/m*(np.exp(m*(x1 - x0)) - 1)

        elif self.interpolation == 'log-log':
            # Log-log
            m = np.log(y1/y0)/np.log(x1/x0)
            partial_sum[i_low:i_high] = y0/((m + 1)*x0**m)*(
                x1**(m + 1) - x0**(m + 1))

        i_low = i_high

        return np.concatenate(([0.], np.cumsum(partial_sum)))

    def to_hdf5(self, group, name='xys1d'):
        """Write tabulated function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=np.vstack(
            [self.x, self.y]))
        dataset.attrs['type'] = np.string_(type(self).__name__)
        dataset.attrs['interpolation'] = self.interpolation

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate tabulated function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Tabulated1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '"
                             + cls.__name__ + "'")

        x = dataset.value[0, :]
        y = dataset.value[1, :]
        interpolation = dataset.attrs['interpolation']
        return cls(x, y, interpolation)
