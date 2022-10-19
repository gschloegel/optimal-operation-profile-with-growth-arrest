import numpy as np
import pandas as pd
from . import base_model


class Model(base_model.Model):
    """Defines the model with exponential feed in the growth phase, and linear feed in the starvation phase.
    This model is analytically solveable and therefore faster to solve than the other models."""
    def __init__(
        self,
        X0=1.52,
        P0=0,
        V0=0.5,
        V_end=1,
        qm=0.08289379,
        Yxg=98 / 180,
        Ypg=0.838,
        Gf=330,
        phase_profile=[("growth", 0.08, 35)],  # [phase, growth rate, end time]
        # in the starvation phase the glucose feed would indicate this growth rate,
        # this hypothtical growth rate is given, not the actual growth rate, that is 0.
        qP_max=0.00829258,
        km=0.11861185,
    ):
        super().__init__(X0, P0, V0, V_end, phase_profile, qm, Yxg, Ypg, Gf)
        self.qP_max = qP_max
        self.km = km

    def qP(self, mu):
        """Calculates the production rate"""
        c1 = self.Ypg / (2 * self.Yxg)
        c2 = -self.Yxg * self.km + self.Yxg * self.qP_max / self.Ypg
        c3 = 2 * self.Yxg * (self.km + self.qP_max / self.Ypg)
        c4 = self.Yxg**2 * (self.km - self.qP_max / self.Ypg) ** 2
        return c1 * (c2 - mu + (mu**2 + c3 * mu + c4) ** (1 / 2))

    def _calc_X_P(self, phase, mu, qP, t, t0, X0, P0):
        """calcultes biomass and product within a phase,
        the values at the beginning of the phase are required."""
        if phase == "growth":
            X = X0 * np.exp(mu * (t - t0))
            P = P0 + qP * X0 / mu * (np.exp(mu * (t - t0)) - 1)
        else:
            X = X0
            P = P0 + qP * X0 * (t - t0)
        return X, P

    def _calc_switches(self):
        """calculates the switches dateframe, containing values at the end of each phase."""
        switches = list()
        X0 = self.X0
        t0 = 0
        V = self.V0
        P = self.P0
        for phase, mu, t in self.phase_profile:
            qP = self.qP(mu)
            X, P = self._calc_X_P(phase, mu, qP, t, t0, X0, P)
            if phase == "growth":
                V = V + (
                    qP / self.Ypg + mu / self.Yxg + self.qm
                ) * X0 / self.Gf / mu * (np.exp(mu * (t - t0)) - 1)
            else:
                V = V + (qP / self.Ypg + mu / self.Yxg + self.qm) * X0 / self.Gf * (
                    t - t0
                )
            switches.append((t, X, P, V))
            t0 = t
            X0 = X
        self.switches = pd.DataFrame(switches, columns=("t", "X", "P", "V")).set_index(
            "t"
        )
        self.Gf = self.Gf * (V - self.V0) / (self.V_end - self.V0)
        self.switches.V = self.V0 + (self.switches.V - self.V0) * (
            self.V_end - self.V0
        ) / (V - self.V0)

    def calc_X_P_end(self):
        """returns final values as set in the parent class."""
        self._calc_switches()
        return self.switches.X.iloc[-1], self.switches.P.iloc[-1]

    def calc(self, points_per_h=5):
        """calculates a complete plotable output for the model."""
        self._calc_switches()
        X0 = self.X0
        V0 = self.V0
        P0 = self.P0
        t0 = 0
        result = list()
        for (phase, mu, t), (X1, P1, V1) in zip(
            self.phase_profile, self.switches.values
        ):
            t_span = np.linspace(t0, t, points_per_h * int(t - t0))
            for t in t_span:
                qP = self.qP(mu)
                X, P = self._calc_X_P(phase, mu, qP, t, t0, X0, P0)
                qG = qP / self.Ypg + mu / self.Yxg + self.qm
                if phase == "growth":
                    V = V0 + (
                        qP / self.Ypg + mu / self.Yxg + self.qm
                    ) * X0 / self.Gf / mu * (np.exp(mu * (t - t0)) - 1)
                    result.append((t, X, P, V, mu, qP, qG))
                else:
                    V = V0 + (
                        qP / self.Ypg + mu / self.Yxg + self.qm
                    ) * X0 / self.Gf * (t - t0)
                    result.append((t, X, P, V, 0, qP, qG))
            t0 = t
            X0 = X1
            P0 = P1
            V0 = V1
        self.results = pd.DataFrame(
            result, columns=("t", "X", "P", "V", "mu", "qP", "qG")
        ).set_index("t")

    def phase_switches(self):
        """returns switch times as required in the parent class."""
        return tuple(self.switches.index)
