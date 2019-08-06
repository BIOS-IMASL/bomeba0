import numpy as np
from scipy import stats


__all__ = ["Uniform", "Normal", "HalfNormal"]


class Uniform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def rvs(self, draws):
        return stats.uniform(self.lower, self.upper - self.lower).rvs(draws)

    def logp(self, value):
        lower = self.lower
        upper = self.upper
        if value < lower:
            return np.inf
        elif value > upper:
            return np.inf
        else:
            return -np.log(upper - lower)


class Normal:
    def __init__(self, mu=0.0, sd=1.0):
        self.mu = mu
        self.sd = sd
        self.tau = np.power(sd, -2)

    def rvs(self, draws):
        return stats.norm(self.mu, self.sd).rvs(draws)

    def logp(self, value):
        tau = self.tau
        mu = self.mu

        if self.sd <= 0:
            return np.inf
        else:
            return (-tau * (value - mu) ** 2 + np.log(tau / np.pi / 2.0)) / 2.0


class HalfNormal:
    def __init__(self, sd=1.0):
        self.mu = 0.0
        self.sd = sd
        self.tau = np.power(sd, -2.0)

    def rvs(self, draws):
        return np.abs(stats.norm(self.mu, self.sd).rvs(draws))

    def logp(self, value):
        tau = self.tau
        mu = self.mu

        if self.sd <= 0:
            return np.inf
        if self.mu < 0:
            return np.inf
        else:
            return (-tau * (value - mu) ** 2 + np.log(tau / np.pi / 2.0)) / 2.0
