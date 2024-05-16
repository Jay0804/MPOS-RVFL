import numpy as np
from clf_MPOSRVFL import MPOS_RVFL

class para_init:

    def __init__(self, n_class=2, X_pt_source=np.array([[]]), y_pt_source=np.array([[]]), n_anchor=100):
        self.n_class = n_class
        self.X_pt_source = X_pt_source
        self.y_pt_source = y_pt_source
        self.n_anchor = n_anchor

    def get_clf(self, name):
        if name == "clf_MPOSRVFL":
            return MPOS_RVFL(
                        Ne=10,
                        N2=20,
                        enhence_function='sigmoid',
                        reg=1,
                        gamma=0.0001,
                        n_anchor=self.n_anchor,
                        sigma=1)
        raise ValueError("no clf found")
