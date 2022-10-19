import numpy as np
import pandas as pd
from . import base_model
import scipy.integrate


class Model(base_model.Model):
    """Defines the alternative implementation of the model with linear feed
    and production rate based on glucose uptake.
    This model is faster than the implimentation using the ODE solver.
    Feed rate and feed concentrations are constant for each phase and set in the phase profile variable."""
    def __init__(
        self,
        X0=1.52,
        P0=0,
        V0=0.5,
        V_end=1,
        phase_profile=[("growth", 35, 330, 1 / 70)],  # [phase, end time, feed concentration, feed rate]
        qm=0.08289379,
        Yxg=98 / 180,
        Ypg=0.838,
        qP_max=0.00829258,
        km=0.11861185,
        Gf=330,
    ):
        super().__init__(X0, P0, V0, V_end, phase_profile, qm, Yxg, Ypg)
        self.qP_max = qP_max
        self.km = km

    def calc_constants(self):
        """calculates constants required for the analytical formula for the ODE"""
        c1 = self.km / (self.qm * (self.Ypg * self.km - self.qP_max))
        c2 = (
            -self.Ypg**2
            * self.km**3
            / (self.qm * (self.Ypg * self.km - self.qP_max))
            + 2
            * self.Ypg
            * self.km**2
            * self.qP_max
            / (self.qm * (self.Ypg * self.km - self.qP_max))
            + self.Ypg * self.km
            - self.km
            * self.qP_max**2
            / (self.qm * (self.Ypg * self.km - self.qP_max))
            + self.qP_max
        ) / (
            self.Ypg * self.km**2
            - self.Ypg * self.km * self.qm
            - self.km * self.qP_max
            - self.qm * self.qP_max
        )
        c3 = self.qP_max / (
            (self.Ypg * self.km - self.qP_max)
            * (self.Ypg * self.km - self.Ypg * self.qm - self.qP_max)
        )
        c4 = (
            -self.Ypg**2 * self.km**2 * self.qP_max
            + 2 * self.Ypg * self.km * self.qP_max**2
            + (self.Ypg * self.km + self.qP_max)
            * (
                (self.Ypg * self.km - self.qP_max)
                * (self.Ypg * self.km - self.Ypg * self.qm - self.qP_max)
            )
            - self.qP_max**3
        ) / (
            (
                self.Ypg * self.km**2
                - self.Ypg * self.km * self.qm
                - self.km * self.qP_max
                - self.qm * self.qP_max
            )
            * (
                (self.Ypg * self.km - self.qP_max)
                * (self.Ypg * self.km - self.Ypg * self.qm - self.qP_max)
            )
        )
        c5 = self.Yxg / self.Ypg
        return c1, c2, c3, c4, c5

    def f(self, X, t, c1, c2, c3, c4, c5, c6, qf, Gf):
        """defines the equation for the biomass. This equation cannot be solved analytically."""
        return (
            -c1 * np.log(-c2 * Gf * qf - X)
            - c3 * np.log(c4 * Gf * qf + X)
            - c5 * t
            - c6
        )

    def calc_root(self, X, T, c1, c2, c3, c4, c5, c6, qf, Gf):
        """Root finder for the biomass equation."""
        try:
            sol = scipy.optimize.root_scalar(
                self.f,
                args=(T, c1, c2, c3, c4, c5, c6, qf, Gf),
                bracket=(0, -c2 * Gf * qf * (1 - 1e-14)),
            )
            return sol.root
        except ValueError:
            print(
                self.f(X, T, c1, c2, c3, c4, c5, c6, qf, Gf),
                self.f(-c2 * Gf * qf * (1 - 1e-14), T, c1, c2, c3, c4, c5, c6, qf, Gf),
            )
            print(T)
            print(X)
            raise ValueError

    def calc_X_P_end(self, quad_length=1, quad_order=5):
        """Calculates the final values and builds switch results list,
        containing the results on the end of each phase."""
        c1, c2, c3, c4, c5 = self.calc_constants()
        X = self.X0
        P = self.P0
        t0 = 0
        self.switch_results = list()
        for phase, T, Gf, qf in self.phase_profile:
            if phase == "growth":
                steps = int((T - t0) // quad_length)
                c6 = self.f(X, t0, c1, c2, c3, c4, c5, 0, qf, Gf)
                tq0 = t0
                for tq in np.linspace(t0, T, steps):
                    # using legendre points and weights for integration of qP * X
                    roots, weights = scipy.special.roots_legendre(quad_order)
                    roots = roots * (tq - tq0) / 2 + ((tq + tq0) / 2)
                    weights *= (tq - tq0) / 2
                    values = list()
                    qPs = list()
                    for root in roots:
                        X = self.calc_root(X, root, c1, c2, c3, c4, c5, c6, qf, Gf)
                        values.append(X)
                        qG = Gf * qf / X
                        qPs.append(self.qP_max / (self.km / (qG - self.qm) + 1))
                    P += sum([w * p * x for w, p, x in zip(weights, qPs, values)])
                    tq0 = tq
                X = self.calc_root(X, T, c1, c2, c3, c4, c5, c6, qf, Gf)
            else:
                qG = Gf * qf / X
                qP = self.qP_max / (self.km / (qG - self.qm) + 1)
                P += qP * X * (T - t0)
            self.switch_results.append((T, X, P))
            t0 = T
        return X, P

    def V(self, t):
        """Calculates volume just depending on the feed rates."""
        V = self.V0
        t0 = 0
        for _, T, _, qf in self.truncate_phase_profile(t):
            V += qf * (T - t0)
            t0 = T
        return V

    def calc(self, step_size=0.25):
        """calculates values within the phases to get a complete (plotable) output"""
        self.calc_X_P_end()
        result = list()
        t0 = 0
        X0 = self.X0
        P = self.P0
        V = self.V0
        c1, c2, c3, c4, c5 = self.calc_constants()
        for ((_, X_end, P_end), (phase, T, Gf, qf)) in zip(
            self.switch_results, self.phase_profile
        ):
            if phase == "growth":
                c6 = self.f(X0, t0, c1, c2, c3, c4, c5, 0, qf, Gf)
                for t in np.arange(t0, T + step_size, step_size):
                    X = self.calc_root(X0, t, c1, c2, c3, c4, c5, c6, qf, Gf)
                    qG = Gf * qf / X
                    qP = self.qP_max / (self.km / (qG - self.qm) + 1)
                    P += qP * (X0 + X) / 2 * step_size
                    mu = (
                        self.Yxg
                        / (self.Ypg * X * (Gf * qf + (self.km - self.qm) * X))
                        * (
                            Gf**2 * self.Ypg * qf**2
                            + (
                                -self.Ypg * self.km * self.qm
                                + self.Ypg * self.qm**2
                                + self.qm * self.qP_max
                            )
                            * X**2
                            + (
                                Gf * self.Ypg * self.km * qf
                                - 2 * Gf * self.Ypg * qf * self.qm
                                - Gf * qf * self.qP_max
                            )
                            * X
                        )
                    )
                    V += qf * step_size
                    result.append((t, X, qP, qG, P, V, mu))
                    X0 = X
            else:
                qG = Gf * qf / X
                qP = self.qP_max / (self.km / (qG - self.qm) + 1)
                mu = 0
                result.append((t0, X, qP, qG, P, V, mu))
                P += qP * X * (T - t0)
                V += qf * (T - t0)
                result.append((T, X, qP, qG, P, V, mu))
            X0 = X_end
            P = P_end
            t0 = T

        self.results = pd.DataFrame(
            result, columns=("t", "X", "qP", "qG", "P", "V", "mu")
        ).set_index("t")

    def phase_switches(self):
        """Returns the phase switching times in accordence with the parent classe."""
        return [t for _, t, _, _ in self.phase_profile]
