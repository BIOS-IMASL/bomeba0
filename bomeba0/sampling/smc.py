import numpy as np
import multiprocessing as mp
from .smc_utils import (
    _initial_population,
    _calc_covariance,
    _tune,
    _calc_beta,
    _metrop_kernel,
    _MultivariateNormalProposal,
    _cpu_count,
)

__all__ = ["smc"]


def smc(
    prior_logp,
    prior,
    likelihood_logp,
    draws=1000,
    n_steps=25,
    parallel=True,
    cores=None,
    scaling=1.0,
    p_acc_rate=0.99,
    tune_scaling=True,
    tune_steps=True,
    threshold=0.5,
    progressbar=False,
):
    """
    Sequential Monte Carlo sampling

    Parameters
    ----------
    prior: function
        The prior distribution.
    likelihood : function
        The likelihood distribution.
    draws : int
        The number of samples to draw from the posterior (i.e. last stage). And
        also the number of independent Markov Chains. Defaults to 5000.
    step : :class:`SMC`
        SMC initialization object
    cores : int
        Number of CPU cores to use. Multiprocessing is used when cores > 1.


    Notes:

    SMC works by moving from successive stages. At each stage the inverse temperature \beta is
    increased a little bit (starting from 0 up to 1). When \beta = 0 we have the prior distribution
    and when \beta =1 we have the posterior distribution. So in more general terms we are always
    computing samples from a tempered posterior that we can write as:

    p(\theta \mid y)_{\beta} = p(y \mid \theta)^{\beta} p(\theta)

    A summary of the algorithm is:

    1. Initialize \beta at zero and stage at zero.
    2. Generate N samples S_{\beta} from the prior (because when \beta = 0 the tempered posterior is
     the prior).
    3. Increase \beta in order to make the effective sample size equals some predefined value
    (we use N*t, where t is 0.5 by default).
    4. Compute a set of N importance weights W. The weights are computed as the ratio of the
    likelihoods of a sample at stage i+1 and stage i.
    5. Obtain S_{w} by re-sampling according to W.
    6. Use W to compute the covariance for the proposal distribution.
    7. For stages other than 0 use the acceptance rate from the previous stage to estimate
    the scaling of the proposal distribution and n_steps.
    8. Run N Metropolis chains (each one of length n_steps), starting each one from a different
    sample in S_{w}.
    9. Repeat from step 3 until \beta \ge 1.  10. The final result is a collection of N samples
    from the posterior.


    References
    ----------
    .. [Minson2013] Minson, S. E. and Simons, M. and Beck, J. L., (2013),
        Bayesian inversion for finite fault earthquake source models I- Theory
        and algorithm.  Geophysical Journal International, 2013, 194(3),
        pp.1701-1726, `link
        <https://gji.oxfordjournals.org/content/194/3/1701.full>`__

    .. [Ching2007] Ching, J. and Chen, Y. (2007).
        Transitional Markov Chain Monte Carlo Method for Bayesian Model
        Updating, Model Class Selection, and Model Averaging. J. Eng. Mech.,
        10.1061/(ASCE)0733-9399(2007)133:7(816), 816-832. `link
        <http://ascelibrary.org/doi/abs/10.1061/%28ASCE%290733-9399
        %282007%29133:7%28816%29>`__
    """
    if cores is None:
        cores = _cpu_count()

    max_steps = n_steps
    p_acc_rate = 1 - p_acc_rate
    accepted = 0
    acc_rate = 1.0
    proposed = draws * n_steps
    stage = 0
    beta = 0
    marginal_likelihood = 1

    posterior = _initial_population(draws, prior)

    print("Sample initial stage: ...")

    while beta < 1:
        # compute plausibility weights (measure fitness)
        if parallel and cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(likelihood_logp, [(sample,) for sample in posterior])
        else:
            results = [likelihood_logp(sample) for sample in posterior]

        likelihoods = np.array(results)
        beta, old_beta, weights, sj = _calc_beta(beta, likelihoods, threshold)
        marginal_likelihood *= sj
        # resample based on plausibility weights (selection)
        resampling_indexes = np.random.choice(np.arange(draws), size=draws, p=weights)
        posterior = posterior[resampling_indexes]
        likelihoods = likelihoods[resampling_indexes]

        # compute proposal distribution based on weights
        covariance = _calc_covariance(posterior, weights)
        proposal = _MultivariateNormalProposal(covariance)

        # compute scaling (optional) and number of Markov chains steps (optional), based on the
        # acceptance rate of the previous stage
        if (tune_scaling or tune_steps) and stage > 0:
            _tune(
                acc_rate,
                proposed,
                tune_scaling,
                tune_steps,
                scaling,
                n_steps,
                max_steps,
                p_acc_rate,
            )

        print("Stage: {:d} Beta: {:.3f} Steps: {:d}".format(stage, beta, n_steps))
        # Apply Metropolis kernel (mutation)
        proposed = draws * n_steps
        priors = np.array([prior_logp(sample) for sample in posterior]).squeeze()
        tempered_logp = priors + likelihoods * beta
        deltas = proposal(n_steps) * scaling

        parameters = (
            proposal,
            scaling,
            accepted,
            n_steps,
            prior_logp,
            likelihood_logp,
            beta,
        )
        if parallel and cores > 1:
            pool = mp.Pool(processes=cores)
            results = pool.starmap(
                _metrop_kernel,
                [
                    (posterior[draw], tempered_logp[draw], *parameters)
                    for draw in range(draws)
                ],
            )
        else:
            results = [
                _metrop_kernel(posterior[draw], tempered_logp[draw], *parameters)
                for draw in range(draws)
            ]

        posterior, acc_list = zip(*results)
        posterior = np.array(posterior)
        acc_rate = sum(acc_list) / proposed
        stage += 1

    return posterior
