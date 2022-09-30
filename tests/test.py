import numpy as np
import AutoEMA as ae

def test_example_data_loading():
    # Load data
    frf, f = ae.load_example()
    print(np.shape(frf))
    print(np.shape(f))
    assert np.shape(frf) == 5, "Error loading data (frf)"
    assert np.shape(f) == 1, "Error loading data (f)"
    return frf, f


def test_example_data_optmodel(frf, f):
    # Init model
    model = ae.OptModel(frf=frf, f_axis=f)
    # Optimize model
    model.optimize(n_init=2, n_iter=2)  # Do more iterations on real data
    return model


def test_example_data_results(model):
    assert model.get_frac() > 0.9, "Error in resulting model"


frf, f = test_example_data_loading()
model = test_example_data_optmodel(frf, f)
test_example_data_results(model)
