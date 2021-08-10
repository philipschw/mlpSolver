# MLP Solver for High-Dimensional Parabolic Semilinear PDEs & BSDEs

Full-history Recursive Multilevel Picard Approximation Solver for High-Dimensional Semilinear Parabolic PDEs.

## Overview

This repository contains the code for simulations concerning the Multilevel Picard approximation algorithm in my master's thesis "Stochastic Approximation Approaches for High-Dimensional Semilinear Parabolic PDE and BSDE". We implemented a gradient-independent as well as a gradient-dependent MLP version each with the possibility to approximate the associated "state" SDE of the PDE under consideration either by an explicit SDE solution or by an SDE approximation scheme. For this purpose, we extended the popular [sdeint](https://github.com/mattja/sdeint) package for integrating SODEs by the package [sdeint2](https://github.com/philipschw/sdeint2) with additional schemes (e.g. further weak approximation schemes). We recommend you to install both packages (`sdeint` and `sdeint2`) directly from their respective repositories.

The derivation of the MLP algortihm can be found in [4]. The implementation of the gradient-independent case is based on [5] and the implementation of the gradient-dependent case on [6]. The implementation for the MLP versions with SDE approximation schemes are inspired by [7].

## Training

```
python main.py --config_path=configs/semilinear_blackscholes.json --exp_name=SemilinearBlackscholes
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve. 
There are nine PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.
* `sample_path`: Directory of samples of one realization of the MLP algorithm in the .npy format.

Explanation of `.py` modules:
* `main.py`: Starts the MLP Approximation Solver. Creates logs.
* `main_memory_saver.py`: Starts the MLP Approximation Solver. Creates logs. Only usable in `advanced` mode (see `advanced` in the config file section below). Saves memory by computing only one realization of the MLP algorithm with pregenerated samples at once. 
* `solver.py`: Incorporates the code for the gradient-depedent and gradient-independent MLP approximation algorithms (parallel computing versions and non parallel computing versions), for the error evaluation, and for the cost evaluation.
* `equation.py`: Base class for defining PDE related function.
* `sampler.py`: Base class for SDE approximation schemes.
* `generateSamples.py`: Generates samples for the advanced mode of the mlpSolver. Auxiliary functions for the advanced mode.

Explanation of the config files:

The config part is splitted into the `eqn_config` and the `eval_config`. All parameters concerning the equation are placed in `eqn_config` and all
parameters concerning the evaluation of the realizations are placed in `eval_config`.
* `_comment`: A comment of the considered example; `str`.
* `eqn_name`: The equation name corresponding to the class object to the equation in `equation.py`; `str`.
* `total_time`: The end time of the time interval of the PDE under consideration; `float`.
* `start_time`: The initial time (the approximation time) of the fixed time-space point of the PDE under consideration; `float`.
* `dim`: The state dimension of the PDE under consideration. If you want to compute different realizations on different dimensions (e.g. dimension 10,100, and 1000), then write `[10,100,100]`; `int` or array of `int`.
* `num_iteration`: The number of iterations of the MLP algorithm `V_{n,n}`. If you want to compute different realizations with different iterations steps (e.g. 1,2,3,4, and 5), then write `[1,2,3,4,5]`; `int` or array of `int`.
* `dim_system`: The dimension of the PDE system under consideration; `int`.
* `gradient_dependent`: Either `true` or `false`. It specifies if you want to use the gradient-dependent or gradient-independent MLP version.
* `time_dist_exponent`: Float value in `(0,1]` to determine the exponent in `numpy.random.power` which is the distribtion of the timepoints in the MLP scheme. If the parameter is equal 1.0 then `numpy.random.power` coincides with `numpy.random.uniform`.
* `num_gridpoints`: The number of gridpoints - 1 which are used in the SDE approximation. For example, the value `[16]` would correspond to 17 gridpoints. If you want to compute different realizations with different number of gridpoints (e.g. 33,65,129), then write `[32,64,128]`; `int` or array of `int`.
* `samplingMethod`: Specifies the sampling method which is used as SDE approximation. Look at `sampler.py` to check which approximation methods are available (and the names). If you want to compute different realizations with different sampling methods (e.g. Euler-Maruyama and Roessler2009 RI5), then write `["EulerMaruyama", "itoRI5"]`; `str` or array of `str`.
* `reference_sol`: Array of reference solutions corresponding to the computed dimensions. If you compute the dimensions [10,100,1000], then you have to specify additionally the solution for each dimension, e.g. `[[11.98736], [14.68754], [17.07766]]`. In PDE systems (e.g. `dim_system` = 2 > 1) you would write `[[0.47656, 2.45639],[0.21889, 4.67698],[0.14272, 6.95539]]`; multidimensional array of `float`.
* `num_realization`: Specifies the number of realizations per fixed dimension `dim`, MLP iteration `num_iteration`, sampling method `samplingMethod`, and number of gridpoints `num_gridpoints`; `int`.
* `multiprocess`: Either `true` or `false`. It specifies if you want to use the parallel programming version of the MLP algorithm (depending on your device's physical cores).
* `verbose`: Either `true` or `false`. It specifies if you want to print logging information during the processing of the MLP algorithm.
* `advanced`: Boolean array of length 8. If `[false,false,false,false,false,false,false,false]`, the advanced mode is disabled. In the advanced mode of the MLP algorithm you pre-generate samples for each dimension `dim`, MLP iteration `num_iteration`, and number of gridpoints `num_gridpoints` and use the same samples for each sampling method (up to the differences in the respective sampling methods) in order to compare the different sampling methods. The entries of the advanced array correspond to the following samples `[timepoints R, Wiener increments dW, Iterated Integrals Ikpw, Iterated Integrals Iwik, Iterated Integrals Imr, Two-Points distribution Itildekp, Three-Points distribution Ihatkp, normal samples Xi]`. Make sure that you activate the corresponding samples by `true` if you want to use the advanced mode and a sampling method specified in `samplingMethod`. Otherwise, this will raise an error.
* `saveSamples`: Either `true` or `false`. If the advanced mode is activated, you can save the generated samples (and reuse them by the `sample_path` flag).


## Problems

`equation.py` and `configs` now support the following problems:

Four examples with gradient-independent nonlinearity in ref [1]:
* `AllenCahnPDE`: Allen-Cahn PDE.
* `SineGordonTypePDE`: Sine-Gordon type PDE.
* `SemilinearBlackScholes`: Semilinear Black-Scholes PDE.
* `SystemSemilinearHeatEquation`: System of semilinear heat equations.

Two examples with gradient-independent nonlinearity in ref [2]:
* `RecursivePricingDefaultRisk`: Recursive Pricing Model with default risk.
* `PricingCreditRisk`: Semilinear PDE for Valuing derivative contracts with counterparty credit risk.

Two examples with gradient-dependent nonlinearity in ref [2]:
* `PricingDifferentInterestRates`: Pricing Problem of an European option in a financial market with different interest rates for borrowing and lending.
* `ExampleExplicitSolution`: An example with an explicit solution.

One example with gradient-independent nonlinearity in ref [3]:
* `SemilinearBlackScholesAmericanOption`: Semlinear Black-Scholes PDE for valuing American options.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. A proper config is needed as well.

Note: The arrangement of the code (i.e. implementation via a `main.py`, `solver.py`, `equation.py`, and `configs`) is inspired by the repository of the deep-learning solver for semilinear parabolic partial differential equations [Deep BSDE Solver](https://github.com/frankhan91/DeepBSDE).

## Dependencies

* `sdeint`
* `sdeint2`

## Reference
[1] Becker, S., Braunwarth, R., Hutzentahler, M., Jentzen, A., and von Wurstemberger, P. Numerical Simulations for Full History Recursive Multilevel Picard Approximations for Systems of High-Dimensional Partial Differential Equations,
<em>Communications in Computational Physics</em>, 28(5), 2109-2138 (2020). [[journal]](http://dx.doi.org/10.4208/cicp.OA-2020-0130) [[arXiv]](https://arxiv.org/abs/2005.10206) <br />
[2] E, W., Hutzenthaler, M., Jentzen, A., Kruse, T. On Multilevel Picard Numerical Approximations for High-Dimensional Nonlinear Parabolic Partial Differential Equations and High-Dimensional Nonlinear Backward Stochastic Differential Equations,
<em>Journal of Scientific Computing</em>, 79(3), 1534-1571 (2019). [[journal]](http://dx.doi.org/10.1007/s10915-018-00903-0) [[arXiv]](https://arxiv.org/abs/1708.03223v1)
<br/>
[3] Benth, F., Karlsen, K., Reikvam, K. A Semilinear Black and Scholes Partial Differential Equation for Valuing American Options,
<em>Finance and Stochastics</em>, 7(3), 277-298 (2003). [[journal]](https://doi.org/10.1007/s007800200091)
<br/>
[4] E, W., Hutzenthaler, M., Jentzen, A., Kruse, T. Multilevel Picard iterations for solving smooth semilinear parabolic heat equations,
1-19 (2019). [[arXiv]](https://arxiv.org/abs/1607.03295v4)
<br/>
[5] Hutzenthaler, M., Jentzen, A., von Wurstemberger, P. Overcoming the curse of dimensionality in the approximative pricing of financial derivatives with default risks,
<em>Electronic Journal of Probability</em>, 25, 1-73 (2020). [[journal]](http://dx.doi.org/10.1214/20-EJP423) [[arXiv]](https://arxiv.org/abs/1903.05985v1)
<br/>
[6] Hutzenthaler, M., Jentzen, A., Kruse, T. Overcoming the curse of dimensionality in the numerical approximation of parabolic partial differential equations with gradient-dependent nonlinearities, 1-33 (2019). [[arXiv]](https://arxiv.org/abs/1912.02571v1)
<br/>
[7] E, W., Hutzenthaler, M., Jentzen, A., Kruse, T. Multilevel Picard approximations for high-dimensional semilinear second-order PDEs with Lipschitz nonlinearities,
1-37 (2020). [[arXiv]](https://arxiv.org/abs/2009.02484v4)
<br/>
