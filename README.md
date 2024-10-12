# SCQuantathon2024
Code for the Xanadu problem: finding spectral gaps for molecules

## Main Learned Points

We found out that approximate VQE is sufficient for the techniques we utilized, and that particle conservation is extremely important for working with molecular hamiltonians.

## VQE & EC

We use Truncated VQE to produce ground states for the $\mathrm{H}_2$ molecule, which can be done in general for any molecule. We've also shown that [Eigenvector Continuation](https://arxiv.org/abs/2406.17037) can be used to extend the results to arbitrary bond lengths, and on fault-tolerant computers, we can take shortcuts with Linear Combination of Unitaries to quickly generate ground states. For NISQ, we extend to Linear Response with Cartan Decomposition.

## Cartan Decomposition

We use the [Cartan Decomposition](https://github.com/ooalshei/Cartan) technique to do fixed-depth time evolution technique to obtain correlation functions. This technique is very suitable for NISQ computers as it is noise-resilient and keeps very low CNOT gate counts, relative other techniques like Trotterization.

## Linear Response

[Linear Response](https://www.nature.com/articles/s41467-024-47729-z) computes correlation functions for our molecular hamiltonians of interest by perturbing our ground state and measuring the response. We can compute the spectral gap directly for the correlation function using Fourier Transforms.
