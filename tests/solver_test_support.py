"""Load the real Python solver without optional FBA/optimizer extensions.

The tests exercise model construction, generated callbacks and SciPy directly.
Cobra, Cython and paropt are unrelated to the non-Cython solver path; keeping
their imports isolated lets these tests run on a small CPU-only installation.
"""

import importlib.util
from pathlib import Path
import sys
import types
from unittest import mock

# Import native numerical extensions before temporarily replacing optional
# package names below. They must remain registered after the import context.
from scipy import integrate


def load_solver(solver_source=None, include_compiled=False):
    package = types.ModuleType("odecell")
    pyximport = types.ModuleType("pyximport")
    pyximport.install = lambda **kwargs: None
    imports = {
        "odecell": package,
        "cobra": types.ModuleType("cobra"),
        "cython": types.ModuleType("cython"),
        "pyximport": pyximport,
    }
    source_dir = Path(__file__).resolve().parents[1] / "odecell"
    with mock.patch.dict(sys.modules, imports):
        names = ("modelbuilder", "solver") + (("compiled",) if include_compiled else ())
        for name in names:
            spec = importlib.util.spec_from_file_location(
                "odecell." + name, source_dir / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            if name == "solver" and solver_source is not None:
                exec(compile(solver_source, str(source_dir / "solver.py"), "exec"), module.__dict__)
            else:
                spec.loader.exec_module(module)
            setattr(package, name, module)
    return tuple(getattr(package, name) for name in names)


def make_chain(modelbuilder, size=2, rate=0.7):
    model = modelbuilder.MetabolicModel()
    model.setVerbosity(0)
    for index in range(size):
        model.addMetabolite("M" + str(index), initVal=1.0)
    for index in range(size - 1):
        reaction = model.addReaction("R" + str(index), "firstOrder")
        model.addSubstrate(reaction, "Sub1", "M" + str(index))
        model.addProduct(reaction, "Prod1", "M" + str(index + 1))
        model.addParameter(reaction, "K", rate)
    return model
