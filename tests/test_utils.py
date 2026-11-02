import numpy as np

from recsys.utils import set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)
