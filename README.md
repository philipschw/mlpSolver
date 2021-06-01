# MLP Solver for High-Dimensional Parabolic Semilinear PDEs and BSDEs
This repository contains the code for simulations in my master's thesis with the working title "Stochastic Approximation Approaches for High-Dimensional Parabolic PDE and BSDE".

Note: The arrangement of the code (i.e. implementation via a `main.py`, `solver.py`, `equation.py`, and `configs`) is inspired by [Deep BSDE Solver](https://github.com/frankhan91/DeepBSDE). In fact the `main.py` is taken with minor changes from this repository.


## Training

```
python main.py --config_path=configs/semilinear_blackscholes.json
```

Command-line flags:

* `config_path`: Config path corresponding to the partial differential equation (PDE) to solve. 
There are two PDEs implemented so far. See [Problems](#problems) section below.
* `exp_name`: Name of numerical experiment, prefix of logging and output.
* `log_dir`: Directory to write logging and output array.


## Problems

`equation.py` and `configs` now support the following problems:

Two examples in ref [1]:
* `SemilinearBlackScholes`: Semilinear Black-Scholes PDE.
* `SystemSemilinearHeatEquation`: System of semilinear heat quations.


New problems can be added very easily. Inherit the class `equation`
in `equation.py` and define the new problem. A proper config is needed as well.


## Reference
[1] Becker, S., Braunwarth, R., Hutzentahler, M., Jentzen, A., and von Wurstemberger, P. Numerical Simulations for Full History Recursive Multilevel Picard Approximations for Systems of High-Dimensional Partial Differential Equations,
<em>Communications in Computational Physics</em>, 28(5), 2109-2138 (2020). [[journal]](http://dx.doi.org/10.4208/cicp.OA-2020-0130) [[arXiv]](https://arxiv.org/abs/1707.02568) <br />
