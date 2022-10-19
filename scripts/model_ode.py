import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import scripts.base_model as base_model


def lin_feed(self, t):
    """simple linear feed rate, to fill reactor in given time."""
    t_end = self.phase_profile[-1][1]
    return 1 / t_end * (self.V_end - self.V0)


def exp_feed_rate(self, t):
    """feed rate ist exponential, calculated from initial parameters"""
    t_end = self.phase_profile[-1][1]
    mu = 0.1
    f0 = (self.V_end - self.V0) * mu / (np.exp(mu * t_end) - 1)
    return f0 * (np.exp(mu * t))


# phase_profile = (("growth", 15), ("starvation", 25), ("growth", 30), ("starvation", 35))
phase_profile = [("growth", 35)]


def qP(X, Gf, qf, phase, qm, q_max, km):
    return q_max * (Gf * qf / X - qm) / (km + (Gf * qf / X - qm))


def Gf(t):
    return 330


class Model(base_model.Model):
    """Models a reversible 2 phase fed-batch process (growth and starvation phase,
    given the desired phase profile.
    The batch phase of this process is not modeled
    assumes the initial glucose concentration is 0 and the start volume is 1
        X0: biomass after batch phase
        P0: product after batch phase
        V0: initial volume
        qf: feed(t) - feed volume over time, given as function dependent on t
        Gf: concentration of glucose in the feed
        phase_profile: lists the phases (growth and starvation) together with end time of the phase
            given as a iterable of the tuples (phase, t_end)
        maintenance: glucose needed for cell maintenance per biomass
        yield_G_B=0.4: biomass yield from glucose
        yield_G_P=0.4: product yield from glucose
        qp: function
        qp_parameter: dictionary
    """

    def __init__(
        self,
        X0=4,
        P0=0.04,
        V0=0.5,
        V_end=1,
        qf=lin_feed,
        Gf=Gf,
        phase_profile=phase_profile,
        qm=0.07,
        Yxg=0.4,
        Ypg=0.4,
        qP=qP,
        qP_parameter={"q_max": 3.629e-18, "km": 8.980e01},
        integration_options=dict(),
    ):

        super().__init__(X0, P0, V0, V_end, phase_profile, qm, Yxg, Ypg, Gf)
        self.qf = qf
        self.qP = qP
        self.qP_parameter = qP_parameter
        self.integration_options = integration_options

    def f(self, t, y):
        """calculates the right side of the differential equations."""
        X = y[0]
        P = y[1]
        G_P = (
            self.qP(X, self.qf(self, t), self.Gf(t), self.qm, self.phase, **self.qP_parameter)
            * X
            / self.Ypg
        )
        if self.phase == "growth":
            G_X = self.qf(self, t) * self.Gf(t) - X * self.qm - G_P
        else:
            G_X = 0
        return np.array((G_X * self.Yxg, G_P * self.Ypg, self.qf(self, t)))

    def calc(self, plot_output=True):
        """solves the differential equation for each phase and calculates the results dataframe."""
        self.t0 = 0
        y = np.array((self.X0, self.P0, self.V0))
        results = list()
        for phase, t_end in self.phase_profile:
            self.phase = phase
            if plot_output:
                res = scipy.integrate.solve_ivp(
                    fun=self.f,
                    t_span=(self.t0, t_end),
                    y0=y,
                    max_step=0.2,
                    **self.integration_options
                )
            else:
                res = scipy.integrate.solve_ivp(
                    fun=self.f,
                    t_span=(self.t0, t_end),
                    y0=y,
                    **self.integration_options
                )
            res = pd.DataFrame(res.y.T, columns=("X", "P", "V"), index=res.t)
            y = np.array(res.iloc[-1:])[0]
            res["phase"] = self.phase
            results.append(res)
            self.t0 = t_end

        self.results = pd.concat(results)
        qf = np.array([self.qf(self, t) for t in self.results.index])
        self.results["qG"] = qf * self.Gf(self.results.index) / self.results.X
        self.results["qP"] = self.qP(
            self.results.X,
            qf,
            self.Gf(self.results.index),
            self.qm,
            self.results.phase,
            **self.qP_parameter
        )
        self.results["mu"] = (
            self.results.qG - self.results.qP / self.Ypg - self.qm
        ) * self.Yxg
        self.results.loc[self.results.phase == "starvation", "mu"] = 0

    def calc_X_P_end(self):
        """returns final values, introduced to confirm with the parent class definition."""
        self.calc(plot_output=False)
        return self.results.X.iloc[-1], self.results.P.iloc[-1]

    def phase_switches(self):
        """accesses the phase swithes as defined in the parent class."""
        return [t for _, t in self.phase_profile]
