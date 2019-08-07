"""
SMC common functions
"""
import numpy as np
from scipy.linalg import cholesky


def _initial_population(draws, priors):
    posterior = np.array([prior.rvs(draws) for prior in priors]).T
    return posterior


def _metrop_kernel(
    q_old,
    old_tempered_logp,
    proposal,
    scaling,
    accepted,
    n_steps,
    prior_logp,
    likelihood_logp,
    beta,
):
    """
    Metropolis kernel
    """
    deltas = proposal(n_steps) * scaling
    for n_step in range(n_steps):
        delta = deltas[n_step]

        q_new = q_old + delta

        new_tempered_logp = prior_logp(q_new) + likelihood_logp(q_new) * beta
        q_old, accept = _metrop_select(
            new_tempered_logp - old_tempered_logp, q_new, q_old
        )

        if accept:
            accepted += 1

            old_tempered_logp = new_tempered_logp

    return q_old, accepted


def _metrop_select(mr, q, q0):
    """Perform rejection/acceptance step for Metropolis class samplers.

    Returns the new sample q if a uniform random number is less than the
    metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.

    Parameters
    ----------
    mr : float, Metropolis acceptance rate
    q : proposed sample
    q0 : current sample

    Returns
    -------
    q or q0
    """
    # Compare acceptance ratio to uniform random number

    if np.isfinite(mr) and np.log(np.random.uniform()) < mr:
        return q, True
    else:
        return q0, False


def _calc_beta(beta, likelihoods, threshold=0.5):
    """
    Calculate next inverse temperature (beta) and importance weights based on
    current beta and tempered likelihood.

    Parameters
    ----------
    beta : float
        tempering parameter of current stage
    likelihoods : numpy array
        likelihoods computed from the model
    threshold : float
        Determines the change of beta from stage to stage, i.e.indirectly the
        number of stages, the higher the value of threshold the higher the
        number of stage. Defaults to 0.5.  It should be between 0 and 1.

    Returns
    -------
    new_beta : float
        tempering parameter of the next stage
    old_beta : float
        tempering parameter of the current stage
    weights : numpy array
        Importance weights (floats)
    sj : float
        Partial marginal likelihood
    """
    low_beta = old_beta = beta
    up_beta = 2.0
    rN = int(len(likelihoods) * threshold)

    while up_beta - low_beta > 1e-6:
        new_beta = (low_beta + up_beta) / 2.0
        weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
        weights = weights_un / np.sum(weights_un)
        ESS = int(1 / np.sum(weights ** 2))
        if ESS == rN:
            break
        elif ESS < rN:
            up_beta = new_beta
        else:
            low_beta = new_beta
    if new_beta >= 1:
        new_beta = 1
    sj = np.exp((new_beta - old_beta) * likelihoods)
    weights_un = np.exp((new_beta - old_beta) * (likelihoods - likelihoods.max()))
    weights = weights_un / np.sum(weights_un)
    return new_beta, old_beta, weights, np.mean(sj)


def _calc_covariance(posterior, weights):
    """
    Calculate trace covariance matrix based on importance weights.
    """
    cov = np.cov(posterior, aweights=weights.ravel(), bias=False, rowvar=0)
    cov = np.atleast_2d(cov)
    cov += 1e-6 * np.eye(cov.shape[0])
    if np.isnan(cov).any() or np.isinf(cov).any():
        raise ValueError('Sample covariances not valid! Likely "draws" is too small!')
    return cov


class _MultivariateNormalProposal:
    def __init__(self, s):
        n, m = s.shape
        if n != m:
            raise ValueError("Covariance matrix is not symmetric.")
        self.n = n
        self.chol = cholesky(s, lower=True)

    def __call__(self, num_draws=None):
        if num_draws is not None:
            b = np.random.randn(self.n, num_draws)
            return np.dot(self.chol, b).T
        else:
            b = np.random.randn(self.n)
            return np.dot(self.chol, b)


def _tune(
    acc_rate,
    proposed,
    tune_scaling,
    tune_steps,
    scaling,
    n_steps,
    max_steps,
    p_acc_rate,
):
    """
    Tune scaling and/or n_steps based on the acceptance rate.

    Parameters
    ----------
    acc_rate: float
        Acceptance rate of the previous stage
    proposed: int
        Total number of proposed steps (draws * n_steps)
    step: SMC step method
    """
    if tune_scaling:
        # a and b after Muto & Beck 2008.
        a = 1 / 9
        b = 8 / 9
        scaling = (a + b * acc_rate) ** 2

    if tune_steps:
        acc_rate = max(1.0 / proposed, acc_rate)
        n_steps = min(
            max_steps, max(2, int(np.log(1 - p_acc_rate) / np.log(1 - acc_rate)))
        )

    return scaling, n_steps


def _cpu_count():
    """Try to guess the number of CPUs in the system.

    We use the number provided by psutil if that is installed.
    If not, we use the number provided by multiprocessing, but assume
    that half of the cpus are only hardware threads and ignore those.
    """
    try:
        import psutil

        cpus = psutil.cpu_count(False)
    except ImportError:
        try:
            cpus = multiprocessing.cpu_count() // 2
        except NotImplementedError:
            cpus = 1
    if cpus is None:
        cpus = 1
    return cpus
