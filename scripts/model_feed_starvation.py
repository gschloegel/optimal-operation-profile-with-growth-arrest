import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
from . import base_model


def lin_feed_rate(self, t):
    """simple linear feed rate, to fill reactor in given time."""
    t_end = self.phase_profile[-1][1]
    return (self.V_end - self.V0) / t_end


def exp_feed_rate(self, t):
    """feed rate ist exponential, calculated from initial parameters"""
    t_end = self.phase_profile[-1][1]
    mu = 0.1
    f0 = (self.V_end - self.V0) * mu / (np.exp(mu * t_end) - 1)
    return f0 * (np.exp(mu * t))


class Model(base_model.Model):
    """Defines the model with constand production rates within the respective phases.
    In this model acumulation of glusose in the medium is modelled."""

    def __init__(
        self,
        X0=1.52,
        P0=0,
        V0=0.5,
        V_end=1,
        qm=0.09207607167280045,
        Yxg=0.7119641297934846,
        Ypg=0.8744883342531962,
        Gf=330,
        phase_profile=[("growth", 35)],
        G0=0,
        mu_max=0.3,  # unrestricted growth rate
        qP_growth=0.00257663763214191,
        # production rate of product per biomass during growth
        qP_starvation=0.0018312561645735484,
        # production rate of product per biomass during starvation
        qf=lin_feed_rate,
    ):
        super().__init__(X0, P0, V0, V_end, phase_profile, qm, Yxg, Ypg, Gf)
        self.G0 = G0
        self.qf = qf
        self.mu_max = mu_max
        self.r_DNA_growth = qP_growth
        self.r_DNA_starvation = qP_starvation

    @property
    def y0(self):
        return np.array((self.X0, self.G0, self.P0, self.V0))

    def feed(self, t, y):
        """Defines the model equation in the growth phase"""
        X = y[0]
        G = y[1]
        G_feed = self.qf(self, t) * self.Gf
        G_maintenance = X * self.qm
        G_product = X * self.r_DNA_growth / self.Ypg
        if G > 1e-3:
            G_used = X * (self.mu_max / self.Yxg) + G_maintenance + G_product
        else:
            G_used = G_feed

        G_growth = G_used - G_maintenance - G_product

        return np.array(
            (
                G_growth * self.Yxg,
                G_feed - G_used,
                G_product * self.Ypg,
                self.qf(self, t),
            )
        )

    def starvation(self, t, y):
        """Defines the model equation in the starvation phase"""
        X = y[0]
        G = y[1]
        G_feed = self.qf(self, t) * self.Gf
        G_maintenance = X * self.qm
        G_product = X * self.r_DNA_starvation / self.Ypg
        if G < 1e3 and G_feed < G_maintenance + G_product:
            G_product = G_feed - G_maintenance

        return np.array(
            (
                0,
                G_feed - G_maintenance - G_product,
                G_product * self.Ypg,
                self.qf(self, t),
            )
        )

    def calc(self, plot_output=True):
        """Solves the differential equations and calculates the dataframe of results.
        plot_output=True sets the minimal step size of the solver to 0.2 to get a output that looks nice when plotted."""

        def no_G(t, y):
            return y[1]

        no_G.terminal = True

        y = self.y0
        results = list()
        t0 = 0

        for phase, t_end in self.phase_profile:
            if phase == "starvation":
                f = self.starvation
            else:
                f = self.feed
            if plot_output:
                res = scipy.integrate.solve_ivp(
                    fun=f, t_span=(t0, t_end), y0=y, max_step=0.2
                )
            else:
                res = scipy.integrate.solve_ivp(fun=f, t_span=(t0, t_end), y0=y)
            res = pd.DataFrame(res.y.T, columns=("X", "G", "P", "V"), index=res.t)
            y = np.array(res.iloc[-1:])[0]
            res["phase"] = phase
            t0 = t_end
            results.append(res)

        self.results = pd.concat(results)
        self.results["qG"] = (
            self.Gf * self.qf(self, self.results.index) / self.results.X
        )
        self.results["mu"] = 0
        self.results["qP"] = 0
        self.results.loc[self.results["phase"] == "growth", "qP"] = self.r_DNA_growth
        self.results.loc[
            self.results["phase"] == "starvation", "qP"
        ] = self.r_DNA_starvation
        self.results.loc[self.results["phase"] == "growth", "mu"] = (
            self.results[self.results.phase == "growth"].qG
            - self.qm
            - self.results[self.results.phase == "growth"].qP / self.Ypg
        ) * self.Yxg

    def calc_X_P_end(self):
        """returns final values, introduced to confirm with the parent class definition."""
        self.calc(plot_output=False)
        return self.results.X.iloc[-1], self.results.P.iloc[-1]

    def phase_switches(self):
        """accesses the phase swithes as defined in the parent class."""
        return [t for ph, t in self.phase_profile]
