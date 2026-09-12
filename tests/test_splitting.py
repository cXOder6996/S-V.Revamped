import pytest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def test_stratified_group_kfold_leakage():
    X = np.zeros((100, 10))
    y = np.array([0]*50 + [1]*50)
    groups = np.array([i // 5 for i in range(100)]) # 20 patients, 5 images each
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]
    
    assert len(set(train_groups) & set(test_groups)) == 0

def test_stratified_group_kfold_distribution():
    X = np.zeros((100, 10))
    y = np.array([0]*50 + [1]*50)
    groups = np.array([i // 5 for i in range(100)]) # 20 patients, 5 images each
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    
    test_y = y[test_idx]
    
    class_0_count = np.sum(test_y == 0)
    class_1_count = np.sum(test_y == 1)
    
    # Ensure test fold class distribution roughly matches the overall 50/50
    assert class_0_count > 0
    assert class_1_count > 0
    assert abs(class_0_count - class_1_count) <= 2
