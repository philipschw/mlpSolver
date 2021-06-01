# MLP Solver for High-Dimensional Parabolic Semilinear PDEs & BSDEs
This repository contains the code for simulations concerning the Multilevel Picard approximation algorithm in my master's thesis "Stochastic Approximation Approaches for High-Dimensional Parabolic PDE and BSDE".


## Training

```
python main.py --config_path=configs/semilinear_blackscholes.json --exp_name=SemilinearBlackscholes
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve. 
There are nine PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.


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

Note: The arrangement of the code (i.e. implementation via a `main.py`, `solver.py`, `equation.py`, and `configs`) is inspired by the repository of the deep-learning solver for semilinear parabolic partial differential equations [Deep BSDE Solver](https://github.com/frankhan91/DeepBSDE). In fact the `main.py` is taken with minor changes from this repository.


## Reference
[1] Becker, S., Braunwarth, R., Hutzentahler, M., Jentzen, A., and von Wurstemberger, P. Numerical Simulations for Full History Recursive Multilevel Picard Approximations for Systems of High-Dimensional Partial Differential Equations,
<em>Communications in Computational Physics</em>, 28(5), 2109-2138 (2020). [[journal]](http://dx.doi.org/10.4208/cicp.OA-2020-0130) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
[2] E, W., Hutzenthaler, M., Jentzen, A., Kruse, T. On Multilevel Picard Numerical Approximations for High-Dimensional Nonlinear Parabolic Partial Differential Equations and High-Dimensional Nonlinear Backward Stochastic Differential Equations,
<em>Journal of Scientific Computing</em>, 79(3), 1534-1571 (2019). [[journal]](http://dx.doi.org/10.1007/s10915-018-00903-0) [[arXiv]](https://arxiv.org/abs/1708.03223v1)
<br/>
[3] Benth, F., Karlsen, K., Reikvam, K. A Semilinear Black and Scholes Partial Differential Equation for Valuing American Options,
<em>Finance and Stochastics</em>, 7(3), 277-298 (2003). [[journal]](https://doi.org/10.1007/s007800200091)