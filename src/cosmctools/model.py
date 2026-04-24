import getdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import harmonic as hm
import harmonic.utils as utils
import yaml
from scipy.stats import genpareto
from flax import serialization
from cosmctools.mcevidence.Cobaya_wrapper import *


class cosmo_model:
    def __init__(self, root, burnin=0.3, nchains=4, block_num=16):
        # base properties
        self.root = root
        self.burnin = burnin
        self.nchains = nchains  # number of MCMC chains in root (change this later to automatically detect)
        self.sampled_params = None  # list of params sampled by MCMC

        # properties for getdist
        self._getdist_samples = None
        self.display_params = None  # list of params to display using getdist (implement later for plots too)
        self.chi2_map = 0
        self.chi2_ml = 0
        # list of dataset names. Add as required.
        self._datasets = [
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
        # properties for MCEvidence
        self._mce_evidence = None
        self._mce_params = None

        # properties for harmonic
        ## properties of the MCMC chains loaded into harmonic
        self._hm_chains = None
        self._block_num = block_num
        self.training = None
        self.inference = None

        ## properties of the flow model and evidence
        self._hm_model = None
        self._hm_evi = None
        self._hm_evidence = None
        self._hm_evidence_error = None

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
        adding derived parameters not tracked by MCMC chains. Add your function here!
        """
        p = self.getdist_samples.getParams()
        # equation of the derived parameter here
        # Omega_b = p.omega_b / (p.H0 / 100) ** 2
        # self.getdist_samples.addDerived(Omega_b, name="Omega_b", label="\\Omega_b")
        if self.getdist_samples.getParamNames().hasParam("q_0") == False:
            q0 = p.Omega_m * 3 / 2 - 1
            self.getdist_samples.addDerived(q0, name="q0", label="q_0")
        if self.getdist_samples.getParamNames().hasParam("rs_d_h") == False:
            rs_d_h = p.rdrag * (p.H0 / 100)
            self.getdist_samples.addDerived(
                rs_d_h, name="rs_d_h", label="r_\mathrm{drag} h"
            )
        if self.getdist_samples.getParamNames().hasParam("H_dS") == False:
            H_dS = p.H0 * np.sqrt(p.Omega_Lambda)
            self.getdist_samples.addDerived(H_dS, name="H_dS", label="\overline{H}")
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
        return param_constraints

    def get_chi2(self, map=True):
        """
        Get -1/2 MAP ln likelihood = chi2_eff for minimised MCMC chains. Requires a .minimum (map=True) or .bestfit (map=False) file from a minimiser sampler in Cobaya.
        """
        bf = self.getdist_samples.getBestFit(max_posterior=map)
        dict = bf.getParamDict()
        self.chi2 = 0
        for data in self._datasets:
            if data in dict:
                self.chi2 += dict[data]
        return self.chi2

    def Delta_chi2(self, alternative_model):
        """
        returning the difference in chi2 between this model and an alternative model. < 0 implies alternative model is favoured, > 0 implies this model is favoured.
        """
        chi2_1 = self.get_chi2()
        chi2_2 = alternative_model.get_chi2()
        delta_chi2 = chi2_2 - chi2_1
        # print(rf"$\Delta \chi^2$: {delta_chi2:.3f}")
        return delta_chi2

    def get_AIC(self):
        """
        Getting the Akaike Information Criterion (AIC) for this model. AIX = chi2_ML + 2k. Requires a .bestfit file from a minimiser sampler in Cobaya.
        """
        if self.sampled_params is None:
            raise ValueError("Please set sampled_params first.")
        else:
            k = len(self.sampled_params)
        aic = self.get_chi2(map=False) + 2 * k
        return aic

    def get_DIC(self, method="alt", user_chi2=None):
        """
        Getting the Deviance Information Criterion (DIC) for this model.
        method refers to the method of obtaining the Bayesian complexity.
        alt: using the alternative definition in BDA3 by Gelman et al. 2013 p_D = 2var(lnL) = 1/2*var(chi2(theta)).
        user: takes the user input of the deviance at the mean. Requires that the user uses the evaluate sampler in Cobaya to obtain the likelihood at the posterior mean.
        read: assumes that one has run the evaluate_at_mean() function which creates a .posterior_mean.1.txt file containing the evaluation of the likelihood at the posterior mean. This reads the chi2 at the posterior mean from this file. Following the definition of p_D in Liddle 2007.
        DIC = chi2(E(theta)) + 2p_D = E(chi2(theta))+p_D
        if not using alt, p_D = E(chi2(theta)) - chi2(E(theta)) which requires the evaluation of the likelihood at the posterior mean.
        """
        mean_chi2 = self.getdist_samples.mean("chi2")
        if method == "alt":
            var_chi2 = self.getdist_samples.var("chi2")
            p_D = 0.5 * var_chi2
            dic = mean_chi2 + p_D
        elif method == "user":
            if user_chi2 is None:
                raise ValueError(
                    "Please provide the chi2 at the posterior mean for this model."
                )
            else:
                chi2_at_mean = user_chi2
            p_D = mean_chi2 - chi2_at_mean
            dic = chi2_at_mean + 2 * p_D
        elif method == "read":
            try:
                with open(f"{self.root}.posterior_mean.1.txt", "r") as f:
                    header = f.readline().strip().lstrip("#").split()
                chain = pd.read_csv(
                    f"{self.root}.posterior_mean.1.txt",
                    sep=r"\s+",
                    comment="#",
                    names=header,
                )
                chi2_at_mean = chain["chi2"]
            except FileNotFoundError:
                raise FileNotFoundError(
                    ".posterior_mean.1.txt file not found. Please run evaluate_at_mean() first to create this file."
                )
            p_D = mean_chi2 - chi2_at_mean
            dic = chi2_at_mean + 2 * p_D
        else:
            raise ValueError(
                "Method not recognised. Please choose alt, liddle or user."
            )
        return dic

    def evaluate_at_mean(self, theory_path=None):
        """
        Function to evaluate the likelihood at the posterior mean. Run this function once after convergence of chains. This will create a .posterior_mean file which can be read to obtain the chi2 at the posterior mean for DIC calculations.

        theory_path: override the `path` entry of the theory block in the original input.yaml (e.g., to point at a local CAMB/CLASS install when the MCMC was run on an HPC with a different path).
        """
        # get the names of sampled parameters
        if self.sampled_params is None:
            raise ValueError("Please set sampled_params first.")
        mean_params = self.getdist_samples.getMeans(self.sampled_params)
        mean_params_dict = {
            param: mean_params[i] for i, param in enumerate(self.sampled_params)
        }

        # reading input.yaml to get the original inputs for the MCMC run
        with open(f"{self.root}.input.yaml") as f:
            input_yaml = yaml.safe_load(f)

        output_loc = self.root + ".posterior_mean"

        if theory_path is not None:
            theory_name = next(iter(input_yaml["theory"]))
            input_yaml["theory"][theory_name]["path"] = theory_path

        input_yaml["sampler"] = {"evaluate": {"override": mean_params_dict}}
        input_yaml["output"] = output_loc
        input_yaml["resume"] = False
        input_yaml["force"] = True

        evaluate_yaml = output_loc + ".yaml"
        with open(evaluate_yaml, "w") as f:
            yaml.safe_dump(input_yaml, f, sort_keys=False)

        # import cobaya and run the evaluation
        try:
            from cobaya import run
        except ImportError:
            raise ImportError(
                "Cobaya is not installed. Please install Cobaya to use this function."
            )
        updated_info, sampler = run(evaluate_yaml)
        sample = sampler.products()["sample"]
        chi2 = float(sample.data["chi2"])
        return chi2

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
        returning the Bayes factor between this model and an alternative model using MCEvidence. > 0 implies alternative model is favoured, < 0 implies this model is favoured.
        """
        evidence1 = self.mcevidence
        evidence2 = alternative_model.mcevidence
        bayes_factor = evidence2 - evidence1
        # print(rf"$\ln\mathcal{{Z}}_\mathrm{{MCE}}$: {bayes_factor:.3f}")
        return bayes_factor

    # %% Learned Harmonic Mean Estimator

    def _load_hm_chains(self):
        chains = hm.Chains(len(self.sampled_params))
        for i in range(self.nchains):
            with open(f"{self.root}.{i + 1}.txt", "r") as f:
                header = f.readline().strip().lstrip("#").split()
            chain = pd.read_csv(
                f"{self.root}.{i + 1}.txt",
                sep=r"\s+",
                comment="#",
                names=header,
            )
            # get log posterior
            chain["logpost"] = -1.0 * chain["minuslogpost"]

            # apply burnin based on cumulative weights
            # commenting out this section as getdist removes e.g. 30% of rows rather than 30% of steps. Therefore, for consistency, it must match what is done by getdist.
            # cumulative_weights = chain["weight"].cumsum()
            # total_steps = cumulative_weights.iloc[-1]
            # if verbose:
            #    print(f"Total steps in chain {i + 1}: {total_steps}")
            # burnin_steps = int(total_steps * self.burnin)
            # start_index = (cumulative_weights >= burnin_steps).idxmax()
            # post_burnin = chain.iloc[start_index:]

            # apply burnin matching getdist
            burnin_index = int(round(len(chain) * self.burnin))
            post_burnin = chain.iloc[burnin_index:].reset_index(drop=True)

            # expand by weight and add to harmonic chains
            weights = post_burnin["weight"].to_numpy().astype(int)
            logpost_array = post_burnin["logpost"].to_numpy()
            chain_array = post_burnin[self.sampled_params].to_numpy()
            expanded_chain = np.repeat(chain_array, weights, axis=0)
            expanded_logpost = np.repeat(logpost_array, weights, axis=0)
            chains.add_chain(expanded_chain, expanded_logpost)
        if self._block_num > self.nchains:
            chains.split_into_blocks(self._block_num)
        elif self._block_num < self.nchains:
            raise ValueError("block_num is less than the number of chains.")
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
        store trained flow model.
        """

        # store architecture
        np.savez(
            f"{self.root}.RQSpline.npz",
            params=np.array(self._mce_params),
            n_layers=self._hm_model.n_layers,
            n_bins=self._hm_model.n_bins,
            offset=self._hm_model.pre_offset,
            amp=self._hm_model.pre_amp,
        )

        # store learned states and weights
        with open(f"{self.root}.RQSpline_state.msgpack", "wb") as f:
            f.write(serialization.to_bytes(self._hm_model.state))

        with open(f"{self.root}.RQSpline_vars.msgpack", "wb") as f:
            f.write(serialization.to_bytes(self._hm_model.variables))
        return self

    def set_temp(self, temp, verbose=False):
        """
        set temperature of the learned distribution for harmonic mean estimation.
        """
        if temp > 1.0:
            raise ValueError(
                "Temperature should be less than or equal to 1.0 for harmonic mean estimation."
            )
        elif self._hm_model is None:
            raise ValueError(
                "Please load or train the model first before setting temperature."
            )
        else:
            self._hm_model.temperature = temp
        if verbose:
            flow_samples = self._hm_model.sample(self.hm_chains.nsamples)
            utils.plot_getdist_compare(self.hm_chains.samples, flow_samples)
            plt.show()
            print("ensure that the concentrated flow is contained within the posterior")
        return self

    def load_model(self, temp=0.9, training_proportion=0.5):
        """
        load trained flow model.
        """
        # load architecture
        data = np.load(f"{self.root}.RQSpline.npz")

        ## setting sampled params
        self.set_sampled_params(data["params"].tolist())

        # load hm_chains if needed
        self.training, self.inference = utils.split_data(
            self.hm_chains, training_proportion=training_proportion
        )

        skeleton = hm.model.RQSplineModel(
            len(self.sampled_params),
            n_layers=int(data["n_layers"]),
            n_bins=int(data["n_bins"]),
            standardize=True,
            temperature=1.0,
        )

        # dummy fit to initialise the model
        skeleton.fit(np.ones((1, len(self.sampled_params))), epochs=1, verbose=False)

        ## setting stardardisation parameters
        skeleton.pre_offset = data["offset"]
        skeleton.pre_amp = data["amp"]

        # load learned states and weights
        with open(f"{self.root}.RQSpline_state.msgpack", "rb") as f:
            skeleton.state = serialization.from_bytes(skeleton.state, f.read())

        with open(f"{self.root}.RQSpline_vars.msgpack", "rb") as f:
            skeleton.variables = serialization.from_bytes(skeleton.variables, f.read())
        self._hm_model = skeleton

        # set temperature
        self.set_temp(temp)
        return self

    def train_model(
        self,
        training_proportion=0.5,
        epochs=15,
        temp=0.9,
        n_layers=3,
        n_bins=128,
        verbose=False,
    ):
        """
        Training the RQSpline model for the Learned Harmonic Mean Estimator using the MCMC chains.
        """
        # split into training and inference sets
        self.training, self.inference = utils.split_data(
            self.hm_chains, training_proportion=training_proportion
        )
        self._hm_model = hm.model.RQSplineModel(
            len(self.sampled_params),
            n_layers=n_layers,
            n_bins=n_bins,
            standardize=True,
            temperature=temp,
        )
        self._hm_model.fit(self.training.samples, epochs=epochs, verbose=verbose)
        if verbose:
            flow_samples = self._hm_model.sample(self.hm_chains.nsamples)
            utils.plot_getdist_compare(self.hm_chains.samples, flow_samples)
            plt.show()
            print("ensure that the concentrated flow is contained within the posterior")
        self.store_model()
        return self

    def _hm_evi_obj(self):
        evi = hm.Evidence(self.inference.nchains, self._hm_model)
        evi.add_chains(self.inference)
        self._hm_evi = evi
        return self

    def get_hm_evidence(self, train_model=False, **kwargs):
        """
        Get -lnZ and (lower, upper) errors on -lnZ
        """
        if self._hm_evidence is None:
            if train_model:
                self.train_model(**kwargs)
            elif self._hm_model is None:
                self.load_model(**kwargs)
            self._hm_evi_obj()
            self._hm_evidence = self._hm_evi.ln_evidence_inv
            self._hm_evidence_error = self._hm_evi.compute_ln_inv_evidence_errors()
        return self._hm_evidence, self._hm_evidence_error

    def check_evidence(self):
        lnk = self._hm_evi.ln_kurtosis
        print("=== Checking Reliability of Evidence ===")
        print(f"ln kurtosis: {lnk:.4f} (>>1 implies evidence may be unreliable)")
        return self

    def get_LHME_bayes_factor(self, alternative_model, **kwargs):
        """
        Returning the Bayes factor between this model and an alternative using Harmonic. > 0 implies the alternative model is favoured, < 0 implies this model is favoured.
        """
        evidence1, evidence1_err = self.get_hm_evidence(**kwargs)
        evidence2, evidence2_err = alternative_model.get_hm_evidence(**kwargs)

        # as harmonic gives -lnZ flip the order of subtraction
        bayes_factor = evidence1 - evidence2

        # flip upper and lower errors to get errors for lnZ
        e1_err_lower, e1_err_upper = abs(evidence1_err[1]), abs(evidence1_err[0])
        e2_err_lower, e2_err_upper = abs(evidence2_err[1]), abs(evidence2_err[0])

        # Upper error of result mixes E2's upper and E1's lower
        err_upper = np.sqrt(e2_err_upper**2 + e1_err_lower**2)

        # Lower error of result mixes E2's lower and E1's upper
        err_lower = np.sqrt(e2_err_lower**2 + e1_err_upper**2)

        # print(
        #    rf"$\ln\mathcal{{Z}}_\mathrm{{LHME}}$: {bayes_factor:.3f} + {err_upper:.3f} / - {err_lower:.3f}"
        # )

        # Returning a tuple containing the Bayes factor and its asymmetric error bounds
        return bayes_factor, err_lower, err_upper

    def check_harmonic_diagnostics(self):
        """
        Checks if the trained flow is strictly contained within the posterior
        by analyzing the importance weights of the inference chains.
        """
        if self.inference is None or self._hm_model is None:
            raise ValueError("Please run get_hm_evidence() or train_model() first.")

        # Evaluate log target density
        log_phi = self._hm_model.predict(self.inference.samples)

        if log_phi.ndim > 1:
            log_phi = log_phi.flatten()

        log_posterior = self.inference.ln_posterior
        if log_posterior.ndim > 1:
            log_posterior = log_posterior.flatten()

        # Calculate the log weights
        log_weights = log_phi - log_posterior

        # Convert to linear weights safely
        weights = np.exp(log_weights - np.max(log_weights))

        # Kish's Effective Sample Size
        ess = np.sum(weights) ** 2 / np.sum(weights**2)
        fractional_ess = ess / len(weights)

        # Pareto-k Diagnostic
        tail_thres = np.percentile(weights, 80)
        tail_weights = weights[weights > tail_thres]

        shifted_tail = tail_weights - tail_thres
        k, *_ = genpareto.fit(shifted_tail, floc=0)

        print("=== Harmonic Estimator Diagnostics ===")
        print(f"Fractional ESS:      {fractional_ess:.4f} (Higher is better)")
        print(f"Pareto-k Diagnostic: {k:.4f} (Less than 0.7 required)")

        if k > 0.7:
            print("Pareto-k > 0.7. Evidence is unreliable.")
        elif k < 0.5:
            print("Pareto-k < 0.5. The Evidence is likely to be reliable.")
        else:
            print("0.5 < Pareto-k < 0.7. May wish to adjust temperature and epochs.")

        return self
