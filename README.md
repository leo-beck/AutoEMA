# Automated Experimental Modal Analysis using Bayesian Optimization (AutoEMA)

Description coming soon

## Installation
Using pip is recommended to install this package. See [this](https://pip.pypa.io/en/stable/installation/) for more information.

```
pip install autoema
```

## Quick Start 
Import package:

``` 
from AutoEMA import AutoEMA as ae 
```


Get exemplary data set: 

``` 
frf, f = ae.load_example() 
```

Initialize model:

``` 
model = ae.OptModel(frf=frf, f_axis=f) 
```

Do the automated modal analysis:

``` 
model.optimize(n_init=2, n_iter=2)  # Do more iterations on real data 
```

Plot the stability diagram:

``` 
model.plot_stability_diagram() 
```

Print information about the resulting model:

``` 
print(model) 
```

Get the results of the modal analysis:

``` 
reconstructed_frf, freq_axis, nat_freqs, damp_ratios, mode_shapes = model.get_results() 
FRAC = model.get_frac() 
```
