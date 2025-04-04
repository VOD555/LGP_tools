import csv
import torch
import numpy as np
from scipy.stats import gaussian_kde
from .utils import differential_evolution

class LGP:
    """
    Local Gaussian Process regression model class.
    """
    def __init__(self, X_train, y_train, device='cpu', kernel_func=None):
        """
        Initialize the GP model.

        Args:
            X_train (array-like): Training input data.
            y_train (array-like): Training output data.
            device (str): 'cpu' or 'cuda'.
            kernel_func (function): Kernel function to use.
        """
        self.device = torch.device(device)
        self.Xd = torch.tensor(X_train, dtype=torch.float32, 
                               device=self.device)
        self.y = torch.tensor(y_train, dtype=torch.float32, 
                              device=self.device)
        self.rnum = self.y.shape[1]
        self.kernel_func = kernel_func if kernel_func else self.se_kernel
        self.optmized_hyperp = None

    @staticmethod
    @torch.no_grad()
    def nearestPD(A):
        """
        Convert a matrix to the nearest positive definite matrix.

        Args:
            A (torch.Tensor): Input square matrix.

        Returns:
            torch.Tensor: Nearest positive definite matrix.
        """
        A_sym = (A + A.t())/2
        if torch.distributions.constraints._PositiveDefinite().check(A_sym):
            return A_sym

        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        A_pos_def = (eigenvectors @ torch.diag(eigenvalues)) @ eigenvectors.t()
        condition = torch.distributions.constraints._PositiveDefinite().check(
            A_pos_def)
        mineigval = torch.min(eigenvalues)
        i = 0
        while not condition:
            A_pos_def = eigenvectors @ torch.diag(
                eigenvalues + 1/(2*10**(5 - i)) - mineigval) @ eigenvectors.t()
            i += 1
            condition = torch.distributions.constraints._PositiveDefinite().check(
                A_pos_def)
        return A_pos_def

    @staticmethod
    def se_kernel(x1, x2, l, width, device):
        """
        Compute the squared exponential (Gaussian) kernel.

        Args:
            x1, x2 (torch.Tensor): Input data points.
            l (torch.Tensor): Length-scale parameters.
            width (float): Kernel amplitude.
            device (torch.device): Computation device.

        Returns:
            torch.Tensor: Kernel matrix.
        """
        x1, x2, l = x1.to(device), x2.to(device), l.to(device)
        scaled_x1, scaled_x2 = x1/l, x2/l
        dist_sq = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return (width**2*torch.exp(-dist_sq/2)).to(device)

    @staticmethod
    def log_prior(mean, std, θ):
        """
        Compute the log prior of the GP hyperparameters.

        Args:
            mean (float): Mean of prior.
            std (float): Std of prior.
            θ (list or tensor): Hyperparameters.

        Returns:
            float: Log prior probability.
        """
        θ = torch.tensor(θ)
        return torch.distributions.Normal(mean, std).log_prob(θ).sum()

    def get_default_prior(self, bounds):
        """
        Estimate prior mean and std from given bounds.

        Args:
            bounds (list of tuple): Log-space bounds of parameters.

        Returns:
            tuple: mean and std tensors.
        """
        bounds = np.array(bounds)
        mean = np.mean(bounds, axis=1)
        std = 2*(bounds[:, 1] - bounds[:, 0])
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(
            std, dtype=torch.float32)

    def hyperp_opt_DE(self, bounds, max_iter=100, pop_size=20, mutation=0.5,
                      crossover=0.7, tol=1e-6, save_path=None):
        """
        Optimize hyperparameters using Differential Evolution.

        Args:
            bounds (list of tuples): Bounds for parameters in log space.
            max_iter (int): Max number of iterations.
            pop_size (int): Population size.
            mutation (float): Mutation factor.
            crossover (float): Crossover probability.
            tol (float): Tolerance for convergence.
        """
        def g_instance(θ):
            θ = torch.tensor(θ, dtype=torch.float32, device=self.device)
            θ = torch.exp(θ)
            return self._g(θ)

        best, best_score = differential_evolution(
            g_instance, bounds, max_iter, pop_size, mutation, crossover, tol,
            save_path=save_path)
        self.optmized_hyperp = np.exp(best)
        print("Optimized hyperparameters:", best,
              "with log marginal likelihood:", best_score)

    def _g(self, θ):
        """
        Compute the log marginal likelihood objective for optimization.

        Args:
            θ (torch.Tensor): Log-transformed hyperparameters.

        Returns:
            float: Log marginal likelihood objective value.
        """
        l, w, σn = θ[:-2], θ[-2], θ[-1]
        Kdd = self.kernel_func(self.Xd, self.Xd, l, w, self.device)
        Kdd += torch.eye(len(self.Xd), device=self.device)*σn
        L = torch.linalg.cholesky(Kdd)
        KddInv = torch.cholesky_inverse(L)

        KddInv_ii = torch.diagonal(KddInv, 0)
        logKddInv_ii = torch.log(KddInv_ii)

        g_val = (1/(2*len(self.Xd)))*torch.sum(
            ((KddInv @ self.y).T/torch.sqrt(KddInv_ii).repeat(
                self.rnum, 1))**2)
        g_val -= (self.rnum/(2*len(self.Xd)))*torch.sum(logKddInv_ii)
        g_val += (self.rnum/2)*np.log(2*np.pi)
        return g_val.item()

    def _local_surrogate(self, Xi, l, width, KddInv, μd, prior):
        """
        Compute the posterior mean estimate.

        Args:
            Xi (torch.Tensor): Test inputs.
            l (torch.Tensor): Length-scales.
            width (float): Kernel amplitude.
            KddInv (torch.Tensor): Inverse of training covariance.
            μd (torch.Tensor): Prior mean of training data.
            prior (function): Prior function.

        Returns:
            torch.Tensor: Posterior mean estimate.
        """
        μ = torch.stack([prior(xi) for xi in Xi]).to(self.device)
        Kid = self.kernel_func(Xi, self.Xd, l, width, self.device)
        return μ + (Kid @ KddInv) @ (self.y - μd)

    def estimator(self, Xi, prior, best=None):
        """
        Predict posterior mean and error using trained GP.

        Args:
            Xi (array-like): Input query points.
            prior (function): Prior function.
            best (list, optional): External hyperparameters.

        Returns:
            tuple: Posterior mean and error covariance matrix.
        """
        Xi = torch.tensor(Xi, dtype=torch.float32, device=self.device)
        if best is not None:
            self.optmized_hyperp = best

        μd = torch.stack([prior(xi) for xi in self.Xd]).float().to(
            self.device)

        arr = torch.tensor(self.optmized_hyperp, dtype=torch.float32,
                          device=self.device)
        l, w, σn = arr[:-2], arr[-2], arr[-1]

        Kdd = self.kernel_func(self.Xd, self.Xd, l, w, self.device)
        Kdd += torch.eye(len(self.Xd), device=self.device)*σn
        Kdd_pd = self.nearestPD(Kdd).to(self.device)
        L = torch.linalg.cholesky(Kdd_pd)
        KddInv = torch.cholesky_inverse(L)

        Kid = self.kernel_func(Xi, self.Xd, l, w, self.device)
        Kii = self.kernel_func(Xi, Xi, l, w, self.device)
        Kii += torch.eye(len(Xi), device=self.device)*σn
        Kdi = self.kernel_func(self.Xd, Xi, l, w, self.device)

        error = (Kii - Kid @ KddInv @ Kdi).to(self.device)
        mean_estimate = self._local_surrogate(
            Xi, l, w, KddInv, μd, prior)
        return mean_estimate.cpu(), error.cpu()
