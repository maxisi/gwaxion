class cached_property(object):    
    '''Computes attribute value and caches it in the instance.
    From the Python Cookbook (Denis Otkidach)
    This decorator allows you to create a property which can be computed once and
    accessed many times. 

    https://stackoverflow.com/questions/7388258/replace-property-for-perfomance-gain
    '''
    def __init__(self, method, name=None):
        # record the unbound-method and the name
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, inst, cls):
        # self: <__main__.cache object at 0xb781340c>
        # inst: <__main__.Foo object at 0xb781348c>
        # cls: <class '__main__.Foo'>       
        if inst is None:
            # instance attribute accessed on class, return self
            # You get here if you write `Foo.bar`
            return self
        # compute, cache and return the instance's attribute value
        result = self.method(inst)
        # setattr redefines the instance's attribute so this doesn't get called again
        setattr(inst, self.name, result)
        return result

import numpy as np

def smooth_data(Z, sigma=2, vmin=-np.inf, vmax=np.inf):
    ## smooth the contours
    from scipy.ndimage.filters import gaussian_filter
    sigma = 2
    # 1. weird gymnastics to deal with missing data
    Z = np.ma.filled(Z, np.nan)
    
    V = np.ma.filled(Z, 0)
    V[V!=V] = 0
    VV = gaussian_filter(V, sigma)
    
    W = 0*Z.copy() + 1
    W[Z!=Z] = 0
    WW = gaussian_filter(W, sigma)
    
    Z_new = VV/WW
    return np.ma.masked_array(Z_new, mask=((Z_new<vmin) | (Z_new>vmax)))
