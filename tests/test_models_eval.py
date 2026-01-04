import numpy as np
import pandas as pd

from stock_success.eval import cross_validate_models, hit_rate
from stock_success.models import instantiate_model, model_registry


def test_hit_rate_directionality():
    y_true = np.array([1, -2, 3, -4])
    y_pred = np.array([0.5, -1, 2, 10])
    assert hit_rate(y_true, y_pred) == 0.75


def test_cross_validate_models_shapes():
    X = pd.DataFrame({"f1": range(30), "f2": range(30, 60)})
    y = pd.Series(np.sin(np.arange(30)))
    models = {"Ridge": instantiate_model("Ridge")}
    results = cross_validate_models(models, X, y, n_splits=3)
    assert not results.empty
    assert set(results.columns) == {"model", "rmse", "mae", "hit_rate", "spearman_ic"}
