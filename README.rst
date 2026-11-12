
ODECell
================

About
-----

**ODECell** provides the infrastructure to create large systems of ODEs, covering multiple cellular subsystems and 
mechanisms, using arbitrary sets rate forms, parameters, and species. The *modelbuilder* module presents 
an interface to define chemical species and concentrations, global parameters (such as pH), rate laws and reactions. 
Partial derivatives of rate laws can be provided for automatic jacobian calculation for every reaction.

The final model object can then be used by the *solver* module, producing a callable object that can be 
used directly by an ODE solver from SciPy.integrate.ode (see Tutorial notebook). 

If parameters were defined with value ranges (upper and lower bounds), the *paropt* module can be used
to optimize their values using multiple approaches (see Tutorial_Optimization notebook). Targets can be set
for both metabolite concentration and reaction fluxes, and the optimization strategy can be either called
through the *paropt* module (using a Differential Evolution implementation from SciPy) or or by an external
optimizer. In the latter case, the *paropt* module provides a "calcObjective" function that returns the cost
for a given parameter set.

Installation instructions (Linux/Unix)
------------------------

1. Create a Python virtual environment.

If you are not already working in a python or conda environment, to create an environment, one can use Anaconda.

Install Anaconda 3 if not already installed on your computer, download and follow the instructions at
https://www.anaconda.com/

To create a new environment named "envname", use

    conda create -n envname -c conda-forge python=3.7.3
    
Activate the environment

    conda activate envname
    
Change the pip version in the environment

    conda install pip==20.3.1

2. Install SUNDIALS in your system so that both headers and libraries are available, as well as its dependencies ( usually provided by development packages for `blas` and `lapack`).

Under Fedora 26 or newer, one can use the example below (as root):

    dnf install sundials-devel blas-devel lapack-devel

For Ubuntu 18.04.1 LTS, one can use (also as root):

    apt-get install libsundials-dev
    
Install SbTab and PyCVODES

    pip install sbtab==0.9.71
    
    conda install -c conda-forge pycvodes=0.14.0
    
Install other dependencies

    conda install -c conda-forge pydantic=1.8.2
    
    conda install -c conda-forge cobra
    
    conda install -c anaconda xlrd==1.2.0
    
    conda install -c anaconda mpi4py
    
3. Install ODECell.

Move into the repository.

    cd odecell

Use `pip` to install the package in your virtual environment.
    
    python -m pip install .
    
This will install all required python packages, and the install ODECell in your activated python virtual environment.

4. Finalize the installation.

Since jupyter extensions are installed as one of the requirements, run the following command to activate the extensions:
    
    jupyter contrib nbextension install --user

Optional cached Cython callbacks
-------------------------------

Repeated metabolic updates can use a compiled callback while continuing to
rebuild the model from the current simulation state::

    from odecell.compiled import buildSolver
    callback = buildSolver(current_model, cache_dir="/path/to/ode_cache")

Pass ``callback`` to the same SciPy integrator used for the Python solver.
``callback.calcFlux(t, y)`` retains the flux interface. State arrays must be
one-dimensional NumPy float64 arrays. The Minimal_Cell_4DWCM integration selects
this path explicitly with ``--odeBackend cython``; its default remains Python.
The backend does not change the integrator, tolerances, output sample times,
stoichiometry, or reaction laws, and does not use an analytical Jacobian.

Each call reads the freshly built model's effective numeric reaction bindings
and explicit parameters into a separate parameter vector. It does not reuse
old enzyme concentrations, medium values or initial conditions. A private model
copy preserves the original derivative accumulation order. Generated source
preserves numeric literal precision and unary-sign syntax; a sign or structural
change may therefore require a new compiled class. Other literal changes, such
as growth or unit-conversion factors, also change the complete source key.

The first call for a new source/ABI key requires Cython, setuptools, NumPy and
Python headers, and a Unix C compiler. The build explicitly disables fast-math
and floating-point contraction. If Python is configured for a compiler that is
not installed, select an available one, for example ``CC=gcc``. Compilation
failures report the build-log path rather than silently switching backends.

All generated sources, metadata, logs and binaries are placed in the selected
cache directory, so the installed package can be read-only. If ``cache_dir`` is
omitted, the package uses ``$XDG_CACHE_HOME/odecell/cython`` or
``~/.cache/odecell/cython``. Minimal_Cell_4DWCM supplies its output directory's
``ode_cache`` by default and supports an explicit ``--odeCacheDirectory``.
Independent processes build in private temporary directories and atomically
publish complete shared objects; simultaneous first use can duplicate the
compilation work but cannot load a partially written binary.

The key includes the full generated source, Python ABI/version, NumPy/Cython
versions, platform/machine and compiler option strings. It does not fingerprint
the compiler executable or all CPU features. Share a cache only between
compatible toolchain/CPU configurations, and use a different directory after
changing a compiler in place or using native CPU flags on a different machine.

CPU validation and timing
~~~~~~~~~~~~~~~~~~~~~~~~~

The shipped 149-metabolite, 175-reaction Minimal_Cell network was tested for
three one-second updates with changed protein counts, volume and medium values.
All 100 output samples and all reaction fluxes matched the original Python path
bit for bit in this fixture. The checks also enforce state agreement at
``rtol=1e-6, atol=1e-12`` and same-state RHS/flux agreement at
``rtol=1e-12, atol=1e-14``. This result is specific to the tested model and
environment; it is not a universal bitwise guarantee for arbitrary rate forms,
compilers or platforms.

On the test runtime (Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
Cython 3.3.1, libSBML 5.21.2, GCC), median repeated metabolic-update time fell
from 3.972 s to 0.708 s (5.61 times faster). Integration itself fell from 3.150 s
to 0.230 s. A separate uncached run of the same generated model took about 72 s
for its first callback generation/compilation, a one-time cost for that source/ABI.
The final validation run reused that binary. These measurements cover the CPU
metabolic portion, not the full spatial simulation or GPU memory use.

Reproduce the full-network comparison in Minimal_Cell_4DWCM with::

    CC=gcc python benchmarks/benchmark_ode.py --baseline-main-ref 08f5f20 --baseline-odecell-ref b19413f --backend cython --cache-dir /path/to/ode_cache

The JSON result is stored in that repository's
``benchmarks/results/cpu_ode_cython.json``. Run the focused ODECell checks with::

    CC=gcc python -m unittest discover -s tests -v

Compiled tests run real compiler/subprocess checks when Cython and the configured
compiler are available; the interpreted solver checks only require NumPy/SciPy.

