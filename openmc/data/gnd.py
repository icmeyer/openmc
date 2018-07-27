# Development notes:
# Want to devlop some objects to assist in creating datastructures
# that are similar to GND.
from collections import Iterable, Callable
from numbers import Real, Integral

from six import add_metaclass
import numpy as np
from numpy.polynomial.legendre import legval

import openmc.data
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin
from .function import Function1D
from .data import EV_PER_MEV

# ACE file interpolation indicators using GND naming scheme
INTERPOLATION_SCHEME = {1: 'flat', 2: 'lin-lin', 3: 'lin-log',
                        4: 'log-lin', 5: 'log-log'}


def from_ace(ace, idx=0, convert_units=True):
    """ Create an XYs1D or Regions1D object as appropriate from an
    ACE table.

    Parameters
    ----------
    ace : openmc.data.ace.Table
        An ACE table
    idx : int
        Offset to read from in XSS array (default of zero)
    convert_units : bool
        If the abscissa represents energy, indicate whether to convert MeV
        to eV.

    Returns
    -------
    openmc.data.XYs1D or openmc.data.Regions1D

    """

    # Get number of regions and pairs
    n_regions = int(ace.xss[idx])
    n_pairs = int(ace.xss[idx + 1 + 2*n_regions])

    # Get interpolation information
    idx += 1
    if n_regions > 0:
        breakpoints = ace.xss[idx:idx + n_regions].astype(int)
        interp_ints = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)

    # Get (x,y) pairs
    idx += 2*n_regions + 1
    x = ace.xss[idx:idx + n_pairs].copy()
    y = ace.xss[idx+n_pairs : idx+2*n_pairs].copy()

    if convert_units:
        x *= EV_PER_MEV

    if n_regions == 0:
        # 0 regions implies linear-linear interpolation by default
        interpolation = 'lin-lin'
        return XYs1D(x, y, interpolation)

    elif n_regions == 1:
        interpolation = INTERPOLATION_SCHEME[interp_ints[0]]
        return XYs1D(x, y, interpolation)

    else:
        # Change breakpoints from 1-index to 0-index
        breakpoints -= 1
        breakpoints = np.insert(breakpoints, 0, 0)
        regions = []
        for i in range(n_regions):
            print(breakpoints[i], breakpoints[i+1])
            # Subregions are boundary inclusive
            if i is not (n_regions - 1):
                region_x = x[breakpoints[i]:breakpoints[i+1]+1]
                region_y = y[breakpoints[i]:breakpoints[i+1]+1]
            else:
                region_x = x[breakpoints[i]:]
                region_y = y[breakpoints[i]:]
            interpolation = INTERPOLATION_SCHEME[interp_ints[i]]
            regions.append(XYs1D(region_x,region_y,interpolation))

        return Regions1D(regions)


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
    interpolation : str, optional
        Interpolation scheme identification

    Attributes
    ----------
    x : Iterable of float
        Independent variable
    domainMin : float
        Minimum x value
    domainMax : float
        Maximum x value
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
        self.domainMin = min(x)
        self.domainMax = max(x)
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
    def domainMin(self):
        return self._domainMin

    @property
    def domainMax(self):
        return self._domainMax

    @property
    def y(self):
        return self._y

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def n_pairs(self):
        return len(self.x)

    @domainMin.setter
    def domainMin(self, domainMin):
        cv.check_type('domain minimum', domainMin, Real)
        self._domainMin = domainMin

    @domainMax.setter
    def domainMax(self, domainMax):
        cv.check_type('domain maximum', domainMax, Real)
        self._domainMax = domainMax

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


class Legendre(np.polynomial.legendre.Legendre, Function1D):
    """GNDS adds capability for storage of data in this format. May be needed
    in the future.

    """
    def to_hdf5(self, group, name='xy'):
        """Write legendre polynomial function to an HDF5 group

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        dataset = group.create_dataset(name, data=self.coef)
        dataset.attrs['type'] = np.string_(type(self).__name__)

    @classmethod
    def from_hdf5(cls, dataset):
        """Generate function from an HDF5 dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            Dataset to read from

        Returns
        -------
        openmc.data.Function1D
            Function read from dataset

        """
        if dataset.attrs['type'].decode() != cls.__name__:
            raise ValueError("Expected an HDF5 attribute 'type' equal to '"
                             + cls.__name__ + "'")
        return cls(dataset.value)


class Regions1D(Function1D):
    """A one-dimensional tabulated function.

    This class mirrors the Regions1D node from the GNDS format.
    It will contain multiple subnodes of type XYs1D or polynomial1D that
    span its domain.
    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = Regions1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    regions : List of objects type openmc.data.gnd.XYs1D or
              openmc.data.gnd.1DSeries

    Attributes
    ----------
    regions : List of objects type openmc.data.gnd.XYs1D or
              openmc.data.gnd.1DSeries
    domainbreaks : Iterable of float
        Breakpoints for interpolation regions
    domainMin : float
        Minimum x value
    domainMax : float
        Maximum x value
    n_regions : int
        Number of interpolation regions

    """

    def __init__(self, regions_list):
        self.regions = []
        n_regions = len(regions_list)
        domainbreaks = np.zeros([n_regions*2])
        for i in range(n_regions):
            domainRange = [regions_list[i].domainMin,regions_list[i].domainMax]
            print(domainRange)
            domainbreaks[2*i:2*i+2] = domainRange
            self.regions.append(regions_list[i])

        # Check that region is filled and domains are monotonically increasing
        for i in range(n_regions-1):
            if domainbreaks[i+1] != domainbreaks[i+2]:
                raise ValueError('Domain bounds of regions do not match')
            # Eliminate duplicate value
            domainbreaks = np.delete(domainbreaks,(i+2))
            if domainbreaks[i+1] > domainbreaks[i+2]:
                raise ValueError('Domains are not monotonically increasing')

        self.domainbreaks = domainbreaks
        self.domainMin = domainbreaks[0]
        self.domainMax = domainbreaks[-1]
        self.n_regions = n_regions

    def __call__(self,x):
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
        idx = np.searchsorted(self.domainbreaks, x)

        #Loop over interpolation regions
        for k in range(self.n_regions):
            # Get indices for the beginning and ending of this region
            regionBegin = self.domainbreaks[k]
            regionEnd = self.domainbreaks[k+1]

            # Figure out which x values lie within this region
            contained = (x >= regionBegin) & (x < regionEnd)
            xk = x[contained] # Apply mask

            #Fill y-values using the member object
            y[contained] = self.region[0](xk)

        # In some cases, x values might be outside the tabulated region due only
        # to precision, so we check if they're close and set them equal if so.

        y[np.isclose(x,self.x[ 0],atol=1e-14)] = self.regions[ 0](self.domainMin)
        y[np.isclose(x,self.x[-1],atol=1e-14)] = self.regions[-1](self.domainMax)

        return y if iterable else y[0]

    def __len__(self):
        return len(self.regions)

    @property
    def domainMin(self):
        return self._domainMin

    @property
    def domainMax(self):
        return self._domainMax

    @property
    def n_regions(self):
        return len(self.x)

    @property
    def breakpoints(self):
        return self._breakpoints

    @domainMin.setter
    def domainMin(self, domainMin):
        cv.check_type('domain minimum', domainMin, Real)
        self._domainMin = domainMin

    @domainMax.setter
    def domainMax(self, domainMax):
        cv.check_type('domain maximum', domainMax, Real)
        self._domainMax = domainMax

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        cv.check_type('breakpoints', breakpoints, Iterable, Real)
        self._breakpoints = breakpoints

    @n_regions.setter
    def n_regions(self, n_regions):
        cv.check_type('number of regions', n_regions, Real)
        self._n_regions = n_regions

    def to_hdf5(self, group, name='regions1d'):
        """Write Regions1D object to an HDF5 group. Utilizes the `to_hdf5`
        functions of its members to create subgroups.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        name : str
            Name of the dataset to create

        """
        group = group.create_group(name)
        group.attrs['type'] = np.string_(type(self).__name__)
        group.attrs['domainbreaks'] = self.domainbreaks
        for idx, region in enumerate(self.regions):
            region.to_hdf5(group, name = 'region'+str(idx))

