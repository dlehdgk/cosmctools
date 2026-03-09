import getdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import harmonic as hm
import harmonic.utils as utils
import equinox as eqx
import sys
from scipy.stats import genpareto
from flax import serialization
from .MCMC_Evidence.Cobaya_wrapper import *


class cosmo_model:
    def __init__(self, root, burnin=0.3, nchains=4):
        self.root = root
        self.burnin = burnin
        self.nchains = nchains
        self.display_params = None
        self.sampled_params = None

        # properties for getdist
        self._getdist_samples = None
        self.chi2 = 0

        # properties for MCEvidence
        self._mce_evidence = None
        self._mce_params = None

        # properties for harmonic
        self.training = None
        self.inference = None
        self.hm_evidence = None
        self.hm_evidence_error = None
        self._hm_model = None
        self._hm_chains = None

    # %% getdist

    @property
    def getdist_samples(self):
        """
        loading getdist samples object from MCMC chains.
        """
        if self._getdist_samples is None:
            self._getdist_samples = getdist.loadMCSamples(
                self.root, settings={"ignore_rows": self.burnin}
            )
        return self._getdist_samples

    def get_gelman_rubin(self):
        """
        returning the Gelman-Rubin convergence criterion for the MCMC chains.
        """
        return self.getdist_samples.getGelmanRubin()

    def set_display_params(self, params):
        """
        setting the list of parameters to be displayed by getdist.
        """
        self.display_params = params
        return self

    def set_sampled_params(self, params):
        """
        setting the list of parameters that were sampled by the MCMC.
        """
        param_list = [0] * len(params)
        for i in range(len(params)):
            if ":" in params[i]:
                param_list[i] = params[i].split(":")[0]
            elif ";" in params[i]:
                param_list[i] = params[i].split(";")[0]
            else:
                param_list[i] = params[i]

        self.sampled_params = param_list
        self._mce_params = params
        return self

    def add_derived(self):
        """
        adding derived parameters not tracked by MCMC chains.
        """
        p = self.getdist_samples.getParams()
        # equation of the derived parameter here
        Omega_b = p.omega_b / (p.H0 / 100) ** 2
        self.getdist_samples.addDerived(Omega_b, name="Omega_b", label="\\Omega_b")
        return self

    def get_params(self, params=None, sigma=1):
        """
        returning requested parameter constraints.
        """
        # if params not specified, use stored display_params, otherwise use provided params
        if params is not None:
            self.set_display_params(params)
        if self.display_params is None:
            raise ValueError(
                "Please set display_params first or provide params argument."
            )

        param_list = self.display_params
        derived_added = False
        # checking if the requested parameters are in the getdist samples
        for param in param_list:
            if not self.getdist_samples.getParamNames().hasParam(param):
                if not derived_added:
                    self.add_derived()
                    derived_added = True
                if not self.getdist_samples.getParamNames().hasParam(param):
                    raise ValueError(
                        f"Parameter {param} not found in getdist samples. Please check the name or add it as a derived parameter."
                    )
        param_constraints = {
            param: self.getdist_samples.getInlineLatex(param, sigma)
            for param in param_list
        }
        for param in param_list:
            print(f"{param_constraints[param]}")
        return param_constraints

    def get_chi2(self):
        """
        Get MAP ln likelihood = -1/2 chi2_eff for minimised MCMC chains. Requires a .minimum file from a minimiser sampler in Cobaya.
        """
        bf = self.getdist_samples.getBestFit(max_posterior=True)
        dict = bf.getParamDict()
        # add names of chi2 for observations used
        obs_list = [
            "chi2__bao.desi_dr2",
            "chi2__sn.pantheonplus",
            "chi2__planck_2018_lowl.TT",
            "chi2__act_dr6_cmbonly.PlanckActCut",
            "chi2__act_dr6_cmbonly.ACTDR6CMBonly",
            "chi2__act_dr6_lenslike.ACTDR6LensLike",
            "chi2__spt3g_d1_tne",
            "chi2__muse3glike.cobaya.spt3g_2yr_delensed_ee_optimal_pp_muse",
            "chi2__sn.desdovekie",
            "chi2__sn.pantheonplusshoes",
        ]
        self.chi2 = 0
        for data in obs_list:
            if data in dict:
                self.chi2 += dict[data]
        return self.chi2

    def Delta_chi2(self, alternative_model):
        """
        returning the difference in chi2 between this model and an alternative model. > 0 implies alternative model is favoured, < 0 implies this model is favoured.
        """
        chi2_1 = self.get_chi2()
        chi2_2 = alternative_model.get_chi2()
        delta_chi2 = chi2_1 - chi2_2
        print(rf"$\Delta \chi^2$: {delta_chi2:.3f}")
        return delta_chi2

    # %% MCEvidence

    @property
    def mcevidence(self):
        """
        computes an estimate for the Bayesian Evidence from chais using the set self.sampled_params.
        """
        if self._mce_params is None:
            raise ValueError("Please set sampled_params first.")
        if self._mce_evidence is None:
            self._mce_evidence = MCMC_Evidence(
                self.root, self._mce_params, verbose=False, get_results=True
            )
        return self._mce_evidence

    def get_MCE_bayes_factor(self, alternative_model):
        """
        returning the Bayes factor between this model and an alternative model using MCEvidence. < 0 implies alternative model is favoured, > 0 implies this model is favoured.
        """
        evidence1 = self.mcevidence
        evidence2 = alternative_model.mcevidence
        bayes_factor = evidence1 - evidence2
        print(rf"$\ln\mathcal{{Z}}_\mathrm{{MCE}}$: {bayes_factor:.3f}")
        return bayes_factor

    # %% Learned Harmonic Mean Estimator
    def _load_hm_chains(self, verbose=False):
        chains = hm.Chains(len(self.sampled_params))
        for i in range(self.nchains):
            chain = pd.read_csv(self.root + f".{i + 1}.txt", sep=r"\s+")
            chain = pd.read_csv(
                self.root + f".{i + 1}.txt",
                sep=r"\s+",
                header=None,
                skiprows=1,
                names=chain.keys()[1 : len(chain.keys())],
            )
            if verbose:
                print(f"Chain {i + 1} loaded with shape: {chain.shape}")
                print(f"Chain {i + 1} starts with:")
                print(chain.head())
            chain["logpost"] = -1 * chain["minuslogpost"]
            cumulative_weights = chain["weight"].cumsum()
            total_steps = cumulative_weights.iloc[-1]
            if verbose:
                print(f"Total steps in chain {i + 1}: {total_steps}")
            burnin_steps = int(total_steps * self.burnin)
            start_index = (cumulative_weights >= burnin_steps).idxmax()
            post_burnin = chain.iloc[start_index:]
            keep_params = ["weight", "logpost"] + self.sampled_params
            if verbose:
                print(f"Chain {i + 1} keeping only: {keep_params}")
            filtred = post_burnin[keep_params]
            weights = filtred["weight"].to_numpy().astype(int)
            logpost_array = filtred["logpost"].to_numpy()
            chain_array = filtred[self.sampled_params].to_numpy()

            expanded_chain = np.repeat(chain_array, weights, axis=0)
            expanded_logpost = np.repeat(logpost_array, weights, axis=0)
            if verbose:
                print(f"Chain {i + 1}: {len(expanded_chain)} samples after expansion")
            chains.add_chain(expanded_chain, expanded_logpost)
        return chains

    @property
    def hm_chains(self):
        """
        loading MCMC chains for harmonic from Cobaya MCMC chains.
        """
        if self.sampled_params is None:
            raise ValueError("Please set sampled_params first.")

        if self._hm_chains is None:
            self._hm_chains = self._load_hm_chains()
        return self._hm_chains

    def store_model(self):
        """
        store trained model
        """
        np.savez(
            f"{self.root}.RQSpline.npz",
            offset=self._hm_model.pre_offset,
            amp=self._hm_model.pre_amp,
        )
        with open(f"{self.root}.RQSpline_state.msgpack", "wb") as f:
            f.write(serialization.to_bytes(self._hm_model.state))

        with open(f"{self.root}.RQSpline_vars.msgpack", "wb") as f:
            f.write(serialization.to_bytes(self._hm_model.variables))
        return self

    def load_model(self, temp=0.9, training_proportion=0.5):
        """
        load trained model
        """
        self.training, self.inference = utils.split_data(
            self.hm_chains, training_proportion=training_proportion
        )

        skeleton = hm.model.RQSplineModel(
            len(self.sampled_params), standardize=True, temperature=temp
        )

        # dummy fit to initialise the model
        skeleton.fit(self.training.samples[:10], epochs=1, verbose=False)

        data = np.load(f"{self.root}.RQSpline.npz")
        skeleton.pre_offset = data["offset"]
        skeleton.pre_amp = data["amp"]

        with open(f"{self.root}.RQSpline_state.msgpack", "rb") as f:
            skeleton.state = serialization.from_bytes(skeleton.state, f.read())

        with open(f"{self.root}.RQSpline_vars.msgpack", "rb") as f:
            skeleton.variables = serialization.from_bytes(skeleton.variables, f.read())
        self._hm_model = skeleton
        return self

    def train_model(self, training_proportion=0.5, epochs=15, temp=0.9, verbose=False):
        """
        Training the RQSpline model for the Learned Harmonic Mean Estimator using the MCMC chains.
        """
        self.training, self.inference = utils.split_data(
            self.hm_chains, training_proportion=training_proportion
        )
        self._hm_model = hm.model.RQSplineModel(
            len(self.sampled_params), standardize=True, temperature=temp
        )
        self._hm_model.fit(self.training.samples, epochs=epochs, verbose=verbose)
        if verbose:
            flow_samples = self._hm_model.sample(self.hm_chains.nsamples)
            utils.plot_getdist_compare(self.hm_chains.samples, flow_samples)
            plt.show()
            print("ensure that the concentrated flow is contained within the posterior")
        self.store_model()
        return self

    def get_hm_evidence(
        self,
        train_model=False,
        **kwargs,
    ):
        if train_model:
            self.train_model(**kwargs)
        elif self._hm_model is None:
            self.load_model(**kwargs)
        evi = hm.Evidence(self.inference.nchains, self._hm_model)
        evi.add_chains(self.inference)
        self.hm_evidence = evi.ln_evidence_inv
        self.hm_evidence_error = evi.compute_ln_inv_evidence_errors()
        return self.hm_evidence, self.hm_evidence_error

    def get_LMHE_bayes_factor(self, alternative_model, **kwargs):
        """
        returning the Bayes factor between this model and an alternative using Harmonic. < 0 implies the alternative model is favoured, > 0 implies this model is favoured.
        """
        evidence1, evidence1_err = self.get_hm_evidence(**kwargs)
        evidence2, evidence2_err = alternative_model.get_hm_evidence(**kwargs)

        # as harmonic gives -lnZ flip the order of subtraction
        bayes_factor = evidence2 - evidence1

        # Harmonic returns errors as [lower_err, upper_err].
        # We take absolute values to ensure magnitudes are used in quadrature.
        e1_err_lower, e1_err_upper = abs(evidence1_err[0]), abs(evidence1_err[1])
        e2_err_lower, e2_err_upper = abs(evidence2_err[0]), abs(evidence2_err[1])

        # Propagating asymmetric errors for subtraction (R = E2 - E1)
        # Upper error of result mixes E2's upper and E1's lower
        err_upper = np.sqrt(e2_err_upper**2 + e1_err_lower**2)

        # Lower error of result mixes E2's lower and E1's upper
        err_lower = np.sqrt(e2_err_lower**2 + e1_err_upper**2)

        print(
            rf"$\ln\mathcal{{Z}}_\mathrm{{LHME}}$: {bayes_factor:.3f} + {err_upper:.3f} / - {err_lower:.3f}"
        )

        # Returning a tuple containing the Bayes factor and its asymmetric error bounds
        return bayes_factor, err_lower, err_upper

    def check_harmonic_diagnostics(self):
        """
        Objectively checks if the trained flow is strictly contained within the posterior
        by analyzing the importance weights of the inference chains.
        """
        if self.inference is None or self._hm_model is None:
            raise ValueError("Please run get_hm_evidence() or train_model() first.")

        # Calculate the log weights: ln(w_i) = ln(phi_i) - ln(posterior_i)
        # We use the already-stored model and inference chains.
        log_phi = self._hm_model.predict(self.inference.samples)
        log_posterior = self.inference.ln_posterior
        log_weights = log_phi - log_posterior

        # Convert to linear weights safely by subtracting the max (Standard LogSumExp trick)
        weights = np.exp(log_weights - np.max(log_weights))

        # Maximum Weight Fraction
        max_weight_frac = np.max(weights) / np.sum(weights)

        # Kish's Effective Sample Size
        ess = np.sum(weights) ** 2 / np.sum(weights**2)
        fractional_ess = ess / len(weights)

        # Pareto-k Diagnostic
        tail_thres = np.percentile(weights, 80)
        tail_weights = weights[weights > tail_thres]

        k, *_ = genpareto.fit(tail_weights)

        print("=== Harmonic Estimator Diagnostics ===")
        print(f"Max Weight Fraction: {max_weight_frac:.4f} (Should be < 0.01)")
        print(f"Fractional ESS:      {fractional_ess:.4f} (Higher is better)")
        print(f"Pareto-k Diagnostic: {k:.4f}")

        if k > 0.7:
            print(
                "WARNING: Pareto-k > 0.7. The flow is NOT contained. Evidence is unreliable."
            )
        elif k < 0.5:
            print(
                "SUCCESS: Pareto-k < 0.5. The flow is strictly contained. Evidence is reliable."
            )
        else:
            print(
                "NOTICE: 0.5 < Pareto-k < 0.7. Estimate is okay, but variance is high."
            )

        return k, fractional_ess
