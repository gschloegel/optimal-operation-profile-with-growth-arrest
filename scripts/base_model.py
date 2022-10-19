import numpy as np
import pandas as pd


class Model:
    """basic model defining all parameters valid for all models and output functions
    It defenes all other models and acts as reference for functions using Model objects as input."""

    def __init__(
        self,
        X0=1,
        P0=0,
        V0=0.5,
        V_end=1,
        phase_profile=[("growth", 35)],
        qm=0.08289379,
        Yxg=98 / 180,
        Ypg=0.838,
        Gf=330,
    ):
        self.X0 = X0
        self.P0 = P0
        self.V0 = V0
        self.V_end = V_end
        self.phase_profile = phase_profile
        self.qm = qm
        self.Yxg = Yxg
        self.Ypg = Ypg
        self.Gf = Gf
        self.results = None

    def calc_X_P_end(self):
        """calculates the final concentrations of biomass and product
        returns tuple (X, P)"""
        pass

    def calc(self):
        """calculates plotable output and saves in in the result dataframe
        t is the index
        X, P, V, mu, qP, qG"""
        pass

    def phase_switches(self):
        """returns the time points of phase switches"""
        pass
