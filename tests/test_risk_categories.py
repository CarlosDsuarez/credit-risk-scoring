import numpy as np
import pytest
from risk_categories import (
    compute_thresholds,
    assign_risk_category,
    assign_risk_categories,
    validate_categories,
    get_risk_categories_dict,
)


def test_thresholds_are_ordered():
    scores = np.random.default_rng(42).uniform(0.01, 0.50, 300)
    t25, t75 = compute_thresholds(scores)
    assert t25 < t75


def test_assign_low():
    assert assign_risk_category(0.01, 0.05, 0.15) == 'LOW'


def test_assign_medium():
    assert assign_risk_category(0.08, 0.05, 0.15) == 'MEDIUM'


def test_assign_high():
    assert assign_risk_category(0.20, 0.05, 0.15) == 'HIGH'


def test_assign_boundary_low_medium():
    # p == t25 -> MEDIUM
    assert assign_risk_category(0.05, 0.05, 0.15) == 'MEDIUM'


def test_assign_boundary_medium_high():
    # p == t75 -> HIGH
    assert assign_risk_category(0.15, 0.05, 0.15) == 'HIGH'


def test_assign_risk_categories_returns_list():
    scores = np.array([0.01, 0.08, 0.20])
    cats = assign_risk_categories(scores, 0.05, 0.15)
    assert cats == ['LOW', 'MEDIUM', 'HIGH']


def test_validate_categories_returns_stats():
    y_true = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1])
    cats   = ['LOW'] * 4 + ['MEDIUM'] * 3 + ['HIGH'] * 3
    stats  = validate_categories(y_true, cats)
    assert 'LOW'    in stats
    assert 'HIGH'   in stats
    assert 'default_rate' in stats['HIGH']


def test_risk_categories_dict_structure():
    d = get_risk_categories_dict(0.05, 0.15)
    assert set(d.keys()) == {'LOW', 'MEDIUM', 'HIGH'}
    assert d['LOW']['max_pd']    == pytest.approx(0.05)
    assert d['HIGH']['min_pd']   == pytest.approx(0.15)
    assert d['MEDIUM']['color']  == 'yellow'
