#Development notes:
# Want to devlop some objects to assist in creating datastructures
# that are similar to GND. 
#
from collections import Iterable, Callable
from numbers import Real, Integral

from six import add_metaclass
import numpy as np
from numpy.polynomial.legendre import legval

import openmc.data
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin
from openmc.data.function import Function1D
from .data import EV_PER_MEV

#ACE file interpolation indicators using GND naming scheme
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
    #Get number of regions and pairs
    n_regions = int(ace.xss[idx])
    n_pairs = int(ace.xss[idx+1+2*n_regions])

    #Get interpolation information
    idx += 1
    if n_regions == 0:
        # 0 regions implies lin-lin interpolation by default
        interpolation = 'lin-lin'
        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = ace.xss[idx : idx + n_paris].copy()
        y = ace.xss[idx+n_pairs : idx+2*n_pairs].copy()

        if convert_units:
            x *= EV_PER_MEV

        return XYs1D(x,y,interpolation)

    elif n_regions == 1:
        interpolation = ace.xss[idx+n_regions : idx+2*n_regions].astype(int)
        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = ace.xss[idx : idx+n_paris].copy()
        y = ace.xss[idx+n_pairs : idx+2*n_pairs].copy()

        if convert_units:
            x *= EV_PER_MEV

        return XYs1D(x,y,interpolation)

    else:
        pass
        ###FIXME: add construction of Regions1D



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

    @x.setter
    def x(self, x):
        cv.check_type('x values', x, Iterable, Real)
        self._x = x

    @domainMin.setter
    def domainMin(self, domainMin):
        cv.check_type('domain minimum', domainMin, Iterable, Real)
        self._domainMin = domainMin

    @domainMax.setter
    def domainMax(self, domainMax):
        cv.check_type('domain maximum', domainMax, Iterable, Real)
        self._domainMax = domainMax

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

class Series1D(Function1D):
    """A one-dimensional function stored as coeffecients of a polynomial 
    sequence.

    This class mirrors the 1-d series node from the GNDS format. The expansion
    and the coefficients are specified.

    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = Series1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    c : Iterable of float
        coefficient values
    series_type : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).
    series_values : list
        Coefficients of polynomial to be used in evaluation
    domainMin : float, optional
        Minimum domain value
    domainMax : float, optional
        Maximum domain value

    Attributes
    ----------
    c : Iterable of float
        coefficient values
    series_type : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).
    domainMin : float, optional
        Minimum domain value
    domainMax : float, optional
        Maximum domain value
    """
    ###FIXME basically needs to be written

    def __init__(self, seriesType, series_values, domainMin, domainMax):
        self.seriesType = self.seriesType

    def __call__(self, x):
        if series_type == 'legendre':
            
            


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
    regions_list : List of objects type openmc.data.gnd.XYs1D or 
                   openmc.data.gnd.1DSeries 

    Attributes
    ----------
    regions_list : List of objects type openmc.data.gnd.XYs1D or 
                   openmc.data.gnd.1DSeries 
    domainBreaks : Iterable of float
        Breakpoints for interpolation regions
    domainMin : float
        Minimum x value
    domainMax : float
        Maximum x value
    n_regions : int
        Number of interpolation regions

    """
    
    def __init__(self, regions_list):
        n_regions = len(regions_list)
        domainBreaks = np.zeros([nregions*2])
        for i in range(nregions):
            domainRange = [regions_list[i].domainMin,regions_list[i].domainMax]
            domainBreaks[i:i+2] = domainRange
            self.regions[i] = regions_list[i]
        
        #Check that region is filled and domains are monotonically increasing
        for i in range(nregions-1):
            if domainBreaks[i+1] != domainBreaks[i+2]:
                raise ValueError('Domain bounds of regions do not match')
            domainBreaks.remove(i+2) #Eliminate duplicate values while checking
            if domainBreaks[i+1] > domainBreaks[i+2]:
                raise ValueError('Domains are not monotonically increasing')

        self.domainBreaks = domainBreaks
        self.domainMin = domainBreaks[0]
        self.domainMax = domainBreaks[-1]
        self.n_regions = n_regions

    def __call__(self,x):
        #Check if input is array or scalar
        if isinstance(x, Iterable):
            iterable = True
            x = np.array(x)
        else: 
            iterable = False
            x = np.array([x], dtype=float)

        # Create output array
        y = np.zeros_like(x)

        # Get indices for interpolation
        idx = np.searchsorted(self.domainBreaks, x)
        
        #Loop over interpolation regions
        for k in range(self.n_regions):
            # Get indices for the beginning and ending of this region
            regionBegin = self.domainBreaks[k]
            regionEnd = self.domainBreaks[k+1]

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
        cv.check_type('domain minimum', domainMin, Iterable, Real)
        self._domainMin = domainMin

    @domainMax.setter
    def domainMax(self, domainMax):
        cv.check_type('domain maximum', domainMax, Iterable, Real)
        self._domainMax = domainMax

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        cv.check_type('breakpoints', breakpoints, Iterable, Real)
        self._breakpoints = breakpoints

