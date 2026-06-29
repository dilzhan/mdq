from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import Self
import abc
import functools
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from numpy.typing import ArrayLike
from jax import random

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

jax.config.update("jax_enable_x64", True)


@dataclass
class Gaussian:
    mu: Float[Array, "D"]
    Sigma: Float[Array, "D D"]

    @functools.cached_property
    def cov_SVD(self):
        if jnp.isscalar(self.mu):
            return jnp.eye(1), jnp.sqrt(self.Sigma).reshape(1, 1), jnp.eye(1)
        U, S, VH = jnp.linalg.svd(self.Sigma)
        return U, jnp.sqrt(S), VH

    @functools.cached_property
    def logdet(self) -> Float:
        _, S, _ = self.cov_SVD
        return jnp.sum(jnp.log(S))

    @functools.cached_property
    def cholesky(self) -> Float[Array, "D D"]:
        L = jnp.linalg.cholesky(self.Sigma)
        return L

    def sample(self, sample_size: Int, seed: Int) -> Float[Array, "L D D"]:
        L = self.cholesky
        key = random.key(seed)
        z = random.normal(shape=(sample_size, self.mu.shape[0]), key=key)

        return self.mu + z @ L.T

    def log_pdf(self, x: Float[Array, "D"]) -> Float:
        d = self.mu.shape[0]
        return (
            -d / 2 * jnp.log(2 * jnp.pi)
            - 1 / 2 * jnp.linalg.slogdet(self.Sigma)[1]
            - 1 / 2 * ((x - self.mu).T @ jnp.linalg.solve(self.Sigma, x - self.mu))
        )

    def pdf(self, x: Float[Array, "D"]) -> Float:
        return jnp.exp(self.log_pdf(x))

    def cdf(self, x: Float[Array, "D"]) -> Float:
        if jnp.isscalar(self.mu):
            return 1 / 2 * (1 + jax.lax.erf((x - self.mu) / (self.Sigma * jnp.sqrt(2))))
        random_sample = self.sample(seed=47, sample_size=100000)
        return jnp.mean((random_sample <= x).all(axis=1))

    @functools.cached_property
    def precision(self):
        U, S, VH = self.cov_SVD
        return U @ jnp.diag(1 / S) ** 2 @ VH

    @functools.cached_property
    def mp(self):
        return self.precision @ self.mu

    def prec_mult(self, other: Float[Array, "D"]) -> Float[Array, "D"]:
        return self.precision @ other

    def __getitem__(self, i: Int):
        return Gaussian(jnp.atleast_1d(self.mu[i]), jnp.atleast_2d(self.Sigma[i][i]))

    @functools.singledispatchmethod
    def __add__(self, other: Float[Array, "D"]) -> Self:
        return Gaussian(self.mu + other, self.Sigma)

    def __mul__(self, other: Self) -> Self:
        A_inv = self.precision
        B_inv = other.precision
        C = jnp.linalg.inv(A_inv + B_inv)
        c = C @ (self.mp + other.mp)
        return Gaussian(c, C)

    def __constmul__(self, c: Float) -> Self:
        return Gaussian(c * self.mu, c**2 * self.Sigma)

    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        return Gaussian(A @ self.mu, A @ self.Sigma @ A.T)

    def condition(
        self, A: Float[Array, "N D"], y: Float[Array, "N"], Lambda: Float[Array, "N N"]
    ) -> Self:
        A = jnp.asarray(A, dtype=jnp.float64)

        Gram = A @ self.Sigma @ A.T + Lambda

        if jnp.isscalar(Gram):
            mu = self.mu + (self.Sigma @ A.T) @ (y - A @ self.mu) / Gram
            Sigma = self.Sigma - (self.Sigma @ A.T) @ (A @ self.Sigma) / Gram
        else:
            L = jax.scipy.linalg.cho_factor(Gram, lower=True)

            mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
                L, y - A @ self.mu
            )
            Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
                L, A @ self.Sigma
            )

        return Gaussian(mu, Sigma)

    def plot_ts(
        self,
        x: Float[Array, "N D"],
        y: Float[Array, "N"] = None,
        ax: Axes | None = None,
        color="black",
        title=None,
        name=None,
        xlabel="Index / Dimension",
        ylabel="Value",
    ):
        mean = self.mu
        std = jnp.sqrt(jnp.diag(self.Sigma))

        if not isinstance(ax, Axes):
            fig, ax = plt.subplots(figsize=(8, 5))

        if y is not None:
            ax.plot(x, y, label="Data", color="red")

        ax.plot(
            x,
            mean,
            label=(rf"$\mu({name})$" if name is not None else r"$\mu$"),
            color=color,
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.3,
            label=(rf"$\pm \sigma({name})$" if name is not None else r"$\pm \sigma$"),
        )
        ax.fill_between(
            x,
            mean - 2 * std,
            mean + 2 * std,
            color=color,
            alpha=0.1,
            label=(rf"$\pm 2\sigma({name})$" if name is not None else r"$\pm 2\sigma$"),
        )

        samples = self.sample(sample_size=5, seed=47)
        for sample in samples:
            ax.plot(x, sample, color=color, alpha=0.1)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    def fit_online(self, x: Float[Array, "D"], y: Float, sigma: Float) -> Self:
        Sigma = jnp.linalg.inv(self.precision + 1 / sigma**2 * x @ x.T)
        mu = Sigma @ (self.precision @ self.mu + 1 / sigma**2 * x * y)

        return Gaussian(mu=mu, Sigma=Sigma)


@dataclass
class GaussianProcess:
    mu: Callable[[jnp.ndarray], jnp.ndarray]
    k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __call__(self, x):
        return Gaussian(self.mu(x), self.k(x[:, None, :], x[None, :, :]))

    def condition(self, y, X, sigma):
        return ConditionalGaussianProcess(
            self, y, X, Gaussian(jnp.zeros_like(y), sigma * jnp.eye(len(y)))
        )


class ParametricGaussianProcess(GaussianProcess):
    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray], prior: Gaussian):
        self.phi = phi
        self.prior = prior
        super().__init__(self._mean, self._covariance)

    def _mean(self, x):
        x = jnp.asarray(x)
        return self.phi(x) @ self.prior.mu

    def _covariance(self, x1, x2):
        x1 = jnp.asarray(x1)
        x2 = jnp.asarray(x2)
        return self.phi(x1).squeeze() @ self.prior.Sigma @ self.phi(x2).squeeze().T


class ConditionalGaussianProcess(GaussianProcess):
    def __init__(self, prior, y, X, epsilon: Gaussian):
        self.prior = prior
        self.y = jnp.atleast_1d(y)
        self.X = jnp.atleast_2d(X)
        self.epsilon = epsilon
        super().__init__(self._mean, self._covariance)

    @functools.cached_property
    def predictive_covariance(self):
        return self.epsilon.Sigma + self.prior.k(self.X[:, None, :], self.X[None, :, :])

    @functools.cached_property
    def predictive_covariance_cho(self):
        return jax.scipy.linalg.cho_factor(self.predictive_covariance, lower=True)

    @functools.cached_property
    def representer_weights(self):
        return jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.y - self.prior(self.X).mu - self.epsilon.mu,
        )

    def _mean(self, x):
        x = jnp.asarray(x)
        return (
            self.prior(x).mu
            + self.prior.k(x[..., None, :], self.X[None, :, :])
            @ self.representer_weights
        )

    @functools.partial(jnp.vectorize, signature="(d),(d)->()", excluded={0})
    def _covariance(self, a, b):
        return self.prior.k(a, b) - self.prior.k(
            a, self.X
        ) @ jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho, self.prior.k(self.X, b)
        )


@Gaussian.__add__.register
def _add_Gaussians(self, other: Gaussian) -> Gaussian:
    return Gaussian(self.mu + other.mu, self.Sigma + other.Sigma)


class ExponentialFamily(abc.ABC):
    @abc.abstractmethod
    def sufficient_statistics(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """ABACABA"""

    @abc.abstractmethod
    def log_base_measure(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """ABACABA"""

    @abc.abstractmethod
    def log_partition(
        self, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """ABACABA"""

    def log_pdf(
        self, x: ArrayLike | jnp.ndarray, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        x = jnp.asarray(x)
        linear_term = (
            self.sufficient_statistics(x)[..., None, :] @ natural_parameters[..., None]
        )[..., 0, 0]
        return (
            self.log_base_measure(x)
            + linear_term
            - self.log_partition(natural_parameters)
        )

    def conjugate_log_partition(
        self, alpha: ArrayLike | jnp.ndarray, nu: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        raise NotImplementedError()

    def conjugate_prior(self) -> "ConjugateFamily":
        return ConjugateFamily(self)

    def posterior_parameters(
        self,
        prior_natural_parameters: ArrayLike | jnp.ndarray,
        data: ArrayLike | jnp.ndarray,
        /,
    ):
        prior_natural_parameters = jnp.asarray(prior_natural_parameters)

        prior_alpha, prior_nu = (
            prior_natural_parameters[:-1],
            prior_natural_parameters[-1],
        )
        sufficient_statistics = self.sufficient_statistics(data)
        n = sufficient_statistics[..., 0].size

        expected_sufficient_statistics = jnp.sum(
            sufficient_statistics, axis=tuple(range(sufficient_statistics.ndim - 1))
        )

        return jnp.append(prior_alpha + expected_sufficient_statistics, prior_nu + n)


class ConjugateFamily(ExponentialFamily):
    def __init__(self, likelihood: ExponentialFamily) -> None:
        self._likelihood = likelihood

    def sufficient_statistics(self, w: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        return jnp.append(w, -self._likelihood.log_partition(w))

    def log_base_measure(self, w: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        w = jnp.asarray(w)

        return jnp.zeros_like(w[..., 0])

    def log_partition(
        self, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        natural_parameters = jnp.asarray(natural_parameters)

        alpha, nu = natural_parameters[:-1], natural_parameters[-1]
        return self._likelihood.conjugate_log_partition(alpha, nu)

    def unnormalized_log_pdf(
        self, w: ArrayLike | jnp.ndarray, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        return self.sufficient_statistics(w) @ jnp.asarray(natural_parameters)
