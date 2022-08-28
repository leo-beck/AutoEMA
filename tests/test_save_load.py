import pickle
import os
import sys
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src import AutoEMA as ae


def basemodel_save_and_load():
    # Get data
    data = pickle.load(open(os.path.join(os.getcwd(), '..', 'src', 'data', 'simulated_3dof.p'), 'rb'))[0]
    frf = data['FRFs']
    f = data['f_axis']

    # Create model
    m = ae.BaseModel(frf, f)
    m.run()

    # Save and load
    ae.save_model(m, 'bm')
    m_l = ae.load_model('bm')

    # Delete file
    os.remove("bm.p")

    # Get results
    r = m.get_results()
    r2 = m_l.get_results()
    return r, r2

def optmodel_save_and_load():
    # Get data
    data = pickle.load(open(os.path.join(os.getcwd(), '..', 'src', 'data', 'simulated_3dof.p'), 'rb'))[0]
    frf = data['FRFs']
    f = data['f_axis']

    # Create model
    m = ae.OptModel(frf, f, lowest_f=111, highest_f=222, reg=0.012)
    m.optimize(5, 5)

    # Save and load
    ae.save_model(m, 'bm')
    m_l = ae.load_model('bm')

    # Delete file
    os.remove("bm.p")

    # Get results
    r = m.get_results()
    r2 = m_l.get_results()
    return r, r2


def test_basemodel_results():
    r, r2 = basemodel_save_and_load()
    for i in range(len(r)):
        assert np.sum(r[i] != r2[i]) == 0

def test_optmodel_results():
    r, r2 = optmodel_save_and_load()
    for i in range(len(r)):
        assert np.sum(r[i] != r2[i]) == 0
