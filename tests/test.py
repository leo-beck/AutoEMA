import numpy as np
from AutoEMA import AutoEMA as ae

def example_data_loading():
    # Load data
    frf, f = ae.load_example()
    return frf, f


def example_data_optmodel(frf, f):
    # Init model
    model = ae.OptModel(frf=frf, f_axis=f)
    # Optimize model
    model.optimize(n_init=2, n_iter=2)  # Do more iterations on real data
    return model


def example_data_results(model):
    return model.get_frac()


def test_loading():
    frf, f = example_data_loading()
    assert np.shape(frf) == (25, 15001)
    assert len(f) == 15001

def test_optmodel():
    frf, f = example_data_loading()
    model = example_data_optmodel(frf, f)
    frac = example_data_results(model)
    assert frac > 0.9
