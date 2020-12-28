[![Documentation Status](https://readthedocs.org/projects/gwaxion/badge/?version=latest)](https://gwaxion.readthedocs.io/en/latest/?badge=latest)

# gwaxion

This is a package to facilitate computations related to gravitational waves from ultralight-boson condensates around black holes.
The code was primarilty developed in writing [arXiv:1810.03812](https://arxiv.org/abs/1810.03812), but it is of broad applicability.

The primary objects in this package are:
- `BlackHoleBoson`
- `BosonCloud`
Their use, as well as other features, is demonstrated in a set of Jupyter notebooks in the `examples/` directory.

Computations of boson cloud evolution and gravitational-wave emission rely on results from [arXiv:1411.0686](https://arxiv.org/abs/1411.0686) and [arXiv:1706.06311](https://arxiv.org/abs/1706.06311).

## Installation

```
pip install gwaxion
```

## Usage

Here is a barebones example to get some properties of a given black-hole--boson system, for a BH with $M = 50 M_\odot$ and $\chi = 0.9$, and a boson with mass such that $\alpha = 0.2$.

``` python
import gwaxion

# create a black-hole--boson object (scalar boson by default)
bhb = gwaxion.BlackHoleBoson.from_parameters(m_bh=50, chi_bh=0.9, alpha=0.2)

# get the fastest-growing boson level
# (l, m, nr, growth rate in Hz)
bhb.max_growth_rate()
# > (1, 1, 0, 4.175501554995195e-06)

# get the mass of the corresponding cloud after superradiant growth
# as a fraction of the original BH mass
cloud = bhb.best_cloud()
cloud.mass / cloud.bhb_initial.bh.mass
# > 0.066
```

For more examples see the `examples/` directory.

## Credit

This code was developed by [Maximiliano Isi](http://maxisi.me) with important contributions by [Richard Brito](https://richardbrito.weebly.com).

If you make use of this code for your own publications, please cite:
```
@article{Isi:2018pzk,
    author = "Isi, Maximiliano and Sun, Ling and Brito, Richard and Melatos, Andrew",
    title = "{Directed searches for gravitational waves from ultralight bosons}",
    eprint = "1810.03812",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P1800270",
    doi = "10.1103/PhysRevD.99.084042",
    journal = "Phys. Rev. D",
    volume = "99",
    number = "8",
    pages = "084042",
    year = "2019",
    note = "[Erratum: Phys.Rev.D 102, 049901 (2020)]"
}
```

(This might be replaced by a dedicated publication at a later date.)
