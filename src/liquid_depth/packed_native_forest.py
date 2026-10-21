import numpy as np
class PackedForest:

    def __init__(self, model):
        trees = [m.tree_ for m in model.estimators_]
        self.is_classifier = hasattr(model, 'classes_')
        self.classes_ = getattr(model, 'classes_', None)
        self.n_jobs = 1
        self.max_depth = max((t.max_depth for t in trees))
        width = max((t.node_count for t in trees))
        n = len(trees)
        self.feature = np.zeros((n, width), np.int32)
        self.threshold = np.zeros((n, width), np.float64)
        self.left = np.zeros((n, width), np.int32)
        self.right = self.left.copy()
        self.leaf = np.ones((n, width), bool)
        self.value = np.zeros((n, width), np.float64)
        for i, t in enumerate(trees):
            k = t.node_count
            self.feature[i, :k] = np.maximum(t.feature, 0)
            self.threshold[i, :k] = t.threshold
            self.left[i, :k] = t.children_left
            self.right[i, :k] = t.children_right
            self.leaf[i, :k] = t.children_left < 0
            values = t.value[:, 0, :]
            if self.is_classifier:
                self.value[i, :k] = values[:, 1] / values.sum(1)
            else:
                self.value[i, :k] = values[:, 0]
