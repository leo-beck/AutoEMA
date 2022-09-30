from bayes_opt import BayesianOptimization
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sdypy import EMA
import numpy as np
from urllib.request import urlopen
import pickle


class BaseModel:
    def __init__(self, frf: np.ndarray, f_axis: np.ndarray, lowest_f: float = None,
                 highest_f: float = None, params: dict = None):
        """
        :param frf: Frequency response function
        :param f_axis: Frequency vector of FRF
        :param lowest_f: Lowest frequency to use for calculation
        :param highest_f: Highest frequency to use for calculation
        :param params: Dict or None. Dictionary of params (type of every param: float)
                key n_max: Max pole order of least-squares complex frequency-domain estimator (LSCF)
                key err_fn: Max difference of frequency (Hz) to >1 previous order pole (same pole as for err_ceta)
                key err_ceta: Max difference of damping ratio (-) to >1 previous order pole (same pole as for err_fn)
                key max_ceta: Maximum value of damping ratio (-)
                key dist: Distance of each cluster of poles (Hz)
                key min_poles: Minimum amount of poles to classify a cluster as valid. (Rule: min_poles < n_poles/n_max)
                key max_norm: Maximum of euclidean norm to classify a cluster as valid
        """        
        self.frf = np.array(frf)
        assert len(np.shape(self.frf)) == 2, \
            f"Expected frf to be 2-dimensional but it has the shape: {str(np.shape(self.frf))}"
        assert np.count_nonzero(np.isnan(self.frf) + np.isinf(self.frf)) == 0, \
            "frf must not contain NaNs or Infs!"
        self.f_axis = np.array(f_axis)
        assert (len(np.shape(self.f_axis)) == 1), \
            f"Expected f_axis to be 1-dimensional but it has the shape: {str(np.shape(self.f_axis))}"
        assert len(self.f_axis) == np.shape(self.frf)[1], \
            f"Expected f_axis[:] and frf[0,:] to have same len, but it's {len(self.f_axis)} and {np.shape(self.frf)[1]}"
        # Set range of frequency
        if lowest_f is None:
            self.lowest_f = f_axis[0]
            self.lowest_f_arg = 0
        else:
            self.lowest_f = float(lowest_f)
            self.lowest_f_arg = np.argmin(abs(self.lowest_f - self.f_axis))
        if highest_f is None:
            self.highest_f = f_axis[-1]
        else:
            self.highest_f = float(highest_f)
        self.highest_f_arg = np.argmin(abs(self.highest_f - self.f_axis))
        if self.highest_f_arg+1 == len(self.f_axis):
            self.highest_f_arg = len(self.f_axis)
        # Set f_axis and frf according to defined frequency range
        self.f_axis = self.f_axis[:self.highest_f_arg]
        self.frf = self.frf[:, :self.highest_f_arg]
        # Set initial params
        self.params = {"n_max": 80, "err_fn": 0.01, "err_ceta": 0.05, "min_ceta": 0,
                       "max_ceta": 0.2, "dist": 2, "min_poles": 0.3, "max_norm": 0.5}
        self.set_params(params)
        # Initialize variables
        self.sampling_time = 0
        self.all_poles = []
        self.H = np.zeros_like(self.frf)  # rebuilt FRF
        self.nf = []  # natural frequency
        self.dr = []  # damping ratio
        self.ms = []  # mode shapes
        self.ms_normal = []  # normalized mode shape
        self.poles_f = []  # Store the poles' frequency after preprocessing
        self.poles_ceta = []  # Store the poles' damping ratios after preprocessing
        self.poles_f_backup = []  # Store poles after calculating
        self.poles_ceta_backup = []  # Store poles after calculating
        self.final_poles_complex = []  # Store complex form of final poles

    def set_params(self, params: dict):
        """ Overwrites self.params for each key in params
        :param params: Dictionary of some or all params
        :return: -
        """
        if params is not None:
            for key in params:
                if key in self.params:
                    # Every param except n_max is float.
                    if key == 'n_max':
                        self.params[key] = int(params[key])
                    else:
                        self.params[key] = float(params[key])
                else:
                    raise ValueError(f"PARAMETER '{key}' DOESN'T EXIST. Possible Parameter: {list(self.params.keys())}")

    def run(self) -> int:
        """ This function runs the automated modal analysis for the defined params in self.params
        :return: 1 if success, 0 if no poles were found
        """
        # Execute algorithm. Do not calculate poles again if they are already known
        if len(self.poles_f_backup) < self.params["n_max"]:
            self._lscf()
            # Preprocess poles
            self._preprocess_poles()
            self.poles_f_backup = self.poles_f
            self.poles_ceta_backup = self.poles_ceta_backup
        else:
            self.poles_f = self.poles_f_backup[:self.params["n_max"] + 1]
            self.poles_ceta = self.poles_ceta_backup[:self.params["n_max"] + 1]
        # Validate poles
        self._validate_poles()
        # Exit if no poles are valid
        if len(self.valid_poles_fod[0]) == 0:
            print("Warning: No poles found")
            return 1
        # Cluster valid poles
        self._cluster()
        # Validate cluster and extract final poles
        self._validate_cluster()
        # Perform LSFD to get reconstructed FRF and mode shapes
        self._lsfd()
        return 0

    def get_results(self) -> tuple:
        """
        :return: Rebuilt FRF (H), freqeuncy vector (f_axis), natural frequencies (nf),
                 damping ratios (dr) and mode shapes (ms)
        """
        return self.H[:, self.lowest_f_arg:], self.f_axis[self.lowest_f_arg:], self.nf, self.dr, self.ms

    def get_frac(self) -> float:
        """ Frequency response assurance criteria
        :return: int
        """
        # Return Frac of 0 if self.H is 0
        if np.sum(self.H != np.zeros_like(self.H)) == 0:
            return 0
        frf1 = self.H[:, self.lowest_f_arg:]
        frf2 = self.frf[:, self.lowest_f_arg:]
        # Cut to relevant frequency range
        all_scores = []
        for f1, f2 in zip(frf1, frf2):
            num = np.vdot(f1, f2)**2
            den = np.vdot(f1, f1) * np.vdot(f2, f2)
            all_scores.append(num/den)
        frac = np.mean(np.abs(all_scores))
        return float(frac)

    def plot_stability_diagram(self, show_poles: bool = True, show_nf: bool = True, x_lim: tuple = None,
                               y_lim: tuple = None, path_to_save: str = None, exc_type: str = "s"):
        """ Plots a typical stability diagram containing the given FRF, the reconstructed FRF, the validated poles and
        the estimated natural frequencies
        :param show_poles: Plots all valid poles' order against their frequency if True
        :param show_nf: Marks all found natural frequencies with a vertical thin line
        :param x_lim: Boundaries of the plot in x direction
        :param y_lim: Boundaries of the plot in y direction (of FRF)
        :param path_to_save: If a path is defined here, a picture of the plot will be stored. Example: "/path/name.png"
        :param exc_type: Defines the y-axis unit. "a" for acceleration, "v" for velocity and "s" for position
        :return: -
        """
        # Settings
        color_true = 'tab:blue'
        font_size = 14
        pad = 10
        anchor = (1, 0.9)
        if exc_type == "s":
            denominator = 'm'
        elif exc_type == "v":
            denominator = 'm/s'
        elif exc_type == "a":
            denominator = 'm/s^2'
        else:
            raise ValueError("exc_type can either be 'a', 'v' or 's'")
        x_size = 18
        label_spacing = 0.3
        fig_size = (14, 6)
        color_rebuilt = 'orange'
        # Create plot
        fig, ax1 = plt.subplots(figsize=fig_size, sharey=True)
        fig.suptitle("Stability diagram", fontsize=20)
        ax1.set_xlabel('Frequency (Hz)', fontsize=font_size)
        ax1.set_ylabel(r'FRF $( \frac{' + denominator + '}{N} ) $', color=color_true, fontsize=font_size)
        mean_frf = np.mean(np.abs(self.frf[:, self.lowest_f_arg:]), axis=0)
        mean_h = np.mean(np.abs(self.H[:, self.lowest_f_arg:]), axis=0)
        ax1.semilogy(self.f_axis[self.lowest_f_arg:], mean_frf, color=color_true, label=r'FRF', zorder=2)
        ax1.semilogy(self.f_axis[self.lowest_f_arg:], mean_h, color=color_rebuilt, label='FRF reconstructed', zorder=2)
        ax1.tick_params(axis='x', labelsize=font_size, pad=pad)
        if x_lim is not None:
            ax1.set_xlim(x_lim)
        else:
            ax1.set_xlim(self.lowest_f, self.highest_f)
        if y_lim is not None:
            ax1.set_ylim(y_lim)
        else:
            ax1.set_ylim(np.min(mean_frf)/1.3, np.max(mean_frf)*1.3)
        # Plot natural frequencies as vertical lines
        if show_nf:
            ax1.vlines(self.nf, 0, self.params["n_max"], colors='k', zorder=0, linewidth=0.5,
                       label='Natural frequency')
        ax1.set_zorder(2)
        # instantiate a second axes that shares the x-axis
        ax2 = ax1.twinx()
        ax2.set_zorder(2)
        if x_lim is not None:
            ax2.set_xlim(x_lim)
        else:
            ax2.set_xlim(self.lowest_f, self.highest_f)
        color = 'tab:red'
        if show_poles:
            ax2.scatter(self.valid_poles_fod[0], self.valid_poles_fod[1], color=color, marker='x', label='Pole',
                        s=x_size)
            ax2.set_ylabel('Model order', color=color, fontsize=font_size)
            ax2.set_ylim(0, np.max(self.valid_poles_fod[1]))
            ax2.tick_params(axis='y', labelcolor=color, labelsize=font_size)
        else:
            ax2.get_yaxis().set_visible(False)
        ax2.tick_params(axis='x', labelsize=font_size, pad=pad)
        # Add legend
        fig.legend(loc="upper right", bbox_to_anchor=anchor, prop={'size': font_size}, fancybox=True, framealpha=1,
                   labelspacing=label_spacing)
        # Make plot smaller in horizontal direction to have enough space for the legend
        fig.subplots_adjust(right=0.78)
        # Export if path is defined
        if path_to_save is not None:
            plt.savefig(path_to_save, bbox_inches='tight')
        plt.show()
        return fig

    def plot_damping_ratios(self, path_to_save: str = None):
        """ Plots the valid poles' damping ratio against their frequency.
        Based on:
                Scionti M. and Lanslots J.P.: Stabilisation diagrams: Pole identification using fuzzy clustering
                techniques, 2005, https://doi.org/10.1016/j.advengsoft.2005.03.029.
        :param path_to_save: If a path is defined here, a picture of the plot will be stored. Example: "/path/name.png"
        :return:
        """
        title_size = 18
        font_size = 14
        x_size = 12
        plt.figure(figsize=(10, 6))
        plt.title("Damping ratio against frequency", fontsize=title_size)
        plt.xlabel("Frequency (Hz)", fontsize=font_size)
        plt.ylabel("Damping ratio (-)", fontsize=font_size)
        # Plot all poles and the ones that were chosen to model the system
        plt.plot(self.valid_poles_fod[0], self.valid_poles_fod[2], 'rx', label="Valid pole", markersize=x_size)
        plt.plot(self.nf, self.dr, 'bx', label="Selected pole", markersize=x_size)
        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(1.45, 0.9), prop={'size': font_size}, fancybox=True, framealpha=1,
                   labelspacing=0.3)
        # Make plot smaller in horizontal direction to have enough space for the legend
        plt.subplots_adjust(right=0.65)
        # Export if path is defined
        if path_to_save is not None:
            plt.savefig(path_to_save, bbox_inches='tight')
        plt.show()
        
    def __repr__(self):
        info = ""
        info += "Model order: {}\n".format(len(self.nf))
        info += "FRAC: {:.1f}%\n".format(self.get_frac()*100)
        nfs = list(map('{:.1f}'.format, self.nf))
        info += "Natural frequencies: {}\n".format(nfs)
        drs = list(map('{:.3f}'.format, self.dr))
        info += "Damping ratios: {}".format(drs)
        return info

    # Functions for internal use are marked with the underscore (_) prefix
    def _preprocess_poles(self):
        """ Transform complex poles to frequency and damping ratio, check frequency range
        :return: -
        """
        self.poles_f = []
        self.poles_ceta = []
        for i_poles in self.all_poles:
            f, ceta = self._get_f_and_ceta_from_cf(i_poles)
            # Check which poles are in between the selected frequency range and if ceta is positive
            valid_indices = np.array(self.lowest_f <= f) * np.array(ceta > 0)
            self.poles_f.append(list(f[valid_indices]))
            self.poles_ceta.append(list(ceta[valid_indices]))

    def _validate_poles(self):
        """ Saves the frequency, order and damping ratio of all poles that fulfill the criteria
        :return: -
        """
        f_old = self.poles_f[0]
        ceta_old = self.poles_ceta[0]
        val_poles_f = []
        val_poles_o = []
        val_poles_dr = []
        for i_order, (f, ceta) in enumerate(zip(self.poles_f, self.poles_ceta)):
            # Iterate through every pole of current order
            for i_pole in range(len(f)):
                freq_valid = np.abs((f[i_pole]-f_old)/f_old) < self.params["err_fn"]
                ceta_valid = np.abs((ceta[i_pole]-ceta_old)/ceta_old) < self.params["err_ceta"]
                ceta_low = ceta[i_pole] < self.params["max_ceta"]
                # Check if all criteria are fulfilled (at least 1 previous pole has valid ceta and valid freq) and
                # ceta_low is also valid
                if np.sum(freq_valid * ceta_valid) > 0 and ceta_low:
                    val_poles_f.append(float(f[i_pole]))
                    val_poles_o.append(i_order)
                    val_poles_dr.append(float(ceta[i_pole]))
            # Save f, ceta of current model order to compare for next model order
            f_old = f
            ceta_old = ceta
        # Return poles. fod = frequency, order, damping_ratio
        self.valid_poles_fod = np.array([val_poles_f, val_poles_o, val_poles_dr])

    def _cluster(self):
        """ Perform a clustering algorithm to all stable poles
        :return: -
        """
        # If only one pole exists, clustering will raise error
        if len(self.valid_poles_fod[0]) == 1:
            self.labels = np.array([0])
            return True
        x = np.array(self.valid_poles_fod[0]).reshape(-1, 1)  # Only use frequency for clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.params["dist"], affinity='l1',
                                             linkage='average')
        self.labels = clustering.fit_predict(x)

    def _validate_cluster(self):
        """ Validate the created cluster and choose the mean values of damping ratio and frequency as the representing
        pole of the cluster
        :return: -
        """
        self.ncluster = 0
        center_f = []
        center_ceta = []
        for i in list(np.unique(self.labels)):
            poles_in_cluster = np.array([self.valid_poles_fod[0, :][i == self.labels],
                                         self.valid_poles_fod[1, :][i == self.labels],
                                         self.valid_poles_fod[2, :][i == self.labels]])
            n_poles = len(self.valid_poles_fod[0, :][i == self.labels])
            norm = np.linalg.norm(poles_in_cluster[0]-np.mean(poles_in_cluster[0, :])) / len(poles_in_cluster[0])
            # Check if cluster contains enough poles (>min_poles) and the norm is small enough (<= max_norm)
            # else: drop cluster
            if n_poles/self.params["n_max"] >= self.params["min_poles"] \
                    and norm/self.params["dist"] <= self.params["max_norm"]:
                self.ncluster += 1
                center_f.append(float(np.mean(poles_in_cluster[0, :])))
                center_ceta.append(float(np.median(poles_in_cluster[2, :])))
        # Sort
        indices = np.argsort(center_f)
        self.nf = np.array(center_f)[indices]
        self.dr = np.array(center_ceta)[indices]

    def _lscf(self):
        """ Use the algorithm implemented in pyEMA to calculate all poles
        See: https://github.com/ladisk/pyEMA/blob/master/pyEMA/pyEMA.py
        Adaption:   - Make it possible to keep the original length of frf and frequency vector by overwriting them
                      after the initialization of the Model-class
        return -
        """
        m = EMA.Model(frf=self.frf, freq=self.f_axis, lower=self.lowest_f, upper=self.highest_f,
                      pol_order_high=self.params['n_max'])
        # Overwrite to keep the original shape of the frf and frequency axis
        m.frf = np.array(self.frf)
        m.freq = self.f_axis
        m.sampling_time = 1/(2*m.freq[-1])  # Must be calculated again for new frequency axis
        m.get_poles(show_progress=False)
        # Store results
        self.all_poles = m.all_poles
        # Backup alle poles
        self.all_poles_backup = self.all_poles

    def _lsfd(self):
        """ COPY, PASTE AND MODIFY FROM pyEMA.
            See: https://github.com/ladisk/pyEMA/blob/master/pyEMA/pyEMA.py
        Adaption:   - Use nat. freq. and damp. ratio instead of complex poles
                    - Make it possible to keep the original length of frf and frequency vector
        return -
        """

        self.omega = 2 * np.pi * self.f_axis
        poles = self._get_cf_from_f_and_ceta(self.nf, self.dr)  # Use nat. freq. and damp. ratio
        self.final_poles_complex = poles

        lower_ind = np.argmin(np.abs(self.f_axis - self.lowest_f))
        upper_ind = np.argmin(np.abs(self.f_axis - self.highest_f))

        if upper_ind == len(self.f_axis):  # Otherwise last array item is deleted
            upper_ind += 1

        _freq = self.f_axis[lower_ind:upper_ind]
        _FRF_mat = self.frf[:, lower_ind:upper_ind]

        ome = 2 * np.pi * _freq
        M_2 = len(poles)

        def TA_construction(TA_omega):
            len_ome = len(TA_omega)
            if TA_omega[0] == 0:
                TA_omega[0] = 1.e-2

            _ome = TA_omega[:, np.newaxis]

            # Initialization
            TA = np.zeros([2*len_ome, 2*M_2 + 4])

            # Eigenmodes contribution
            TA[:len_ome, 0:2*M_2:2] =    (-np.real(poles))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-np.real(poles))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[len_ome:, 0:2*M_2:2] =    (-(_ome-np.imag(poles)))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-(_ome+np.imag(poles)))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[:len_ome, 1:2*M_2+1:2] =  ((_ome-np.imag(poles)))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (-(_ome+np.imag(poles)))/(np.real(poles)**2+(_ome+np.imag(poles))**2)
            TA[len_ome:, 1:2*M_2+1:2] =  (-np.real(poles))/(np.real(poles)**2+(_ome-np.imag(poles))**2)+\
                                        (np.real(poles))/(np.real(poles)**2+(_ome+np.imag(poles))**2)

            # Lower and upper residuals contribution
            TA[:len_ome, -4] = -1/(TA_omega**2)
            TA[len_ome:, -3] = -1/(TA_omega**2)
            TA[:len_ome, -2] = np.ones(len_ome)
            TA[len_ome:, -1] = np.ones(len_ome)

            return TA

        AT = np.linalg.pinv(TA_construction(ome))
        FRF_r_i = np.concatenate([np.real(_FRF_mat.T),np.imag(_FRF_mat.T)])
        A_LSFD = AT @ FRF_r_i

        self.A = (A_LSFD[0:2*M_2:2, :] + 1.j*A_LSFD[1:2*M_2+1:2, :]).T
        self.LR = A_LSFD[-4, :]+1.j*A_LSFD[-3, :]
        self.UR = A_LSFD[-2, :]+1.j*A_LSFD[-1, :]
        self.poles = poles

        # FRF reconstruction
        _FRF_r_i = TA_construction(self.omega)@A_LSFD
        frf_ = (_FRF_r_i[:len(self.omega), :] + _FRF_r_i[len(self.omega):, :]*1.j).T
        self.H = frf_
        self.ms = self.A

    # Helper functions
    @staticmethod
    def _get_cf_from_f_and_ceta(f: list, ceta: list) -> np.ndarray:
        """ Calculating the complex frequency given the frequency (f) and damping ratios (ceta)
        Based on:
            Lallement, G. and Inman, D.: A Tutorial on Complex Eigenvalues, 1995
        :param f: Frequency vector
        :param ceta: Damping ratio
        :return:
        """
        f_pi = np.array(f, dtype=complex)*2*np.pi
        real = -np.array(ceta)*np.array(f)*2*np.pi
        imag = np.array(np.abs(f_pi**2-real**2), dtype=complex)**0.5
        return np.array(real+1j*imag, dtype=complex)

    @staticmethod
    def _get_f_and_ceta_from_cf(cf: np.ndarray) -> tuple:
        """ Calculate the frequency (f) and damping ratio (ceta) from the complex frequency cf
        Based on:
                    Lallement, G. and Inman, D.: A Tutorial on Complex Eigenvalues, 1995
        :param cf: Complex natural frequency
        """
        f = np.abs(cf) * np.sign(np.imag(cf))
        ceta = -np.real(cf)/f
        f = f / (2 * np.pi)
        return f, ceta


class OptModel(BaseModel):
    def __init__(self, frf: np.ndarray, f_axis: np.ndarray, lowest_f: int = None, highest_f: int = None,
                 order: int = None, show_progress: bool = True, reg: float = 0.01):
        """ This class optimizes the parameter needed for the automated modal analysis by maximizing the models FRAC
        value and penalizing high model orders.

        :param order: The model order to optimize for. If None, the model order is set automatically.
        :param show_progress: Prints the result of every step of the bayesian optimization
        :param reg: Regularization factor.

        See __init__ of AutoEMA for a documentation of the other arguments
        """
        super().__init__(frf, f_axis, lowest_f=lowest_f, highest_f=highest_f, params=None)
        # Store inputs
        self.reg = float(reg)
        if order is not None:
            self.order = float(order)
        else:
            self.order = None
        self.show_progress = bool(show_progress)
        # Define Boundaries
        p_bounds = {
            'n_max': (60, 120),
            'err_fn': (0.001, 0.1),
            'err_ceta': (0.05, 0.2),
            'max_ceta': (0.2, 0.3),
            'dist': (0.4, 4),
            'min_poles': (0.2, 0.6),
            'max_norm': (0.1, 0.8)
        }
        # Calculate poles once with the highest possible n_max!
        self.set_params({"n_max": p_bounds['n_max'][1]})
        # Run once with the highest possible n_max
        self.run()
        # Save these poles
        self.poles_f_backup = self.poles_f
        self.poles_ceta_backup = self.poles_ceta

        # Define Blackbox function
        def black_box_function(**kwargs):
            """ Returns a score for how good the model is based on its FRAC value and model order
            :return: Score. Type: float. Higher=Better.
            """
            # Build param dict from inputs and set them
            self.set_params(kwargs)
            # Calculate rebuilt FRF
            self.run()
            # Get difference of model order or absolute model order
            if self.order is not None:
                diff_order = (self.order - len(self.nf)) ** 2
            else:
                diff_order = len(self.nf)
            # Apply regularization
            score = self.get_frac() - diff_order * self.reg
            return score

        # Initialize Bayesian optimizer
        if self.show_progress:
            verbose = 2
        else:
            verbose = 0
        self.optimizer = BayesianOptimization(f=black_box_function, pbounds=p_bounds,
                                              verbose=verbose, random_state=1)

    def optimize(self, n_init: int = 20, n_iter: int = 20):
        """ This function performs a """
        # Check inputs
        n_init = int(n_init)
        n_iter = int(n_iter)
        # Optimize
        self.optimizer.maximize(init_points=n_init, n_iter=n_iter)
        # Get best result
        self.set_params(self.optimizer.max['params'])
        # Run again with the best params
        self.run()


def load_example():
    """ This function imports an exemplary FRF
    :return: FRF, corresponding frequency vector
    """
    data = pickle.load(urlopen("https://github.com/leo-beck/AutoEMA/raw/main/data/simulated_3dof.p"))[0]
    frf = data['FRFs']
    freq_axis = data['f_axis']
    return frf, freq_axis


def save_model(model, path_and_name: str):
    """ This function saves the model
    :return: -
    """
    if path_and_name[-2] != '.p':
        path_and_name += '.p'
    # Create dict containing all necessary data
    vals = {'model_type': 'BaseModel',
            'frf': model.frf,
            'f': model.f_axis,
            'low': model.lowest_f,
            'high': model.highest_f}
    if str(type(model)).__contains__('Opt'):
        vals['model_type'] = 'OptModel'
        vals['order'] = model.order
        vals['show_progress'] = model.show_progress
        vals['reg'] = model.reg
    pickle.dump([vals, model.params], open(path_and_name, 'wb'))


def load_model(path_and_name: str):
    """ This function loads a model
    :return: BaseModel or OptModel
    """
    if path_and_name[-2] != '.p':
        path_and_name += '.p'
    vals, params = pickle.load(open(path_and_name, 'rb'))
    if vals['model_type'] == 'BaseModel':
        model = BaseModel(frf=vals['frf'], f_axis=vals['f'], lowest_f=vals['low'],
                          highest_f=vals['high'], params=params)
    else:
        model = OptModel(frf=vals['frf'], f_axis=vals['f'], lowest_f=vals['low'], highest_f=vals['high'],
                         order=vals['order'], show_progress=vals['show_progress'], reg=vals['reg'])
        model.set_params(params)
    model.run()
    return model
