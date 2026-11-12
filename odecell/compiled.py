"""Optional, parameterized Cython callbacks for repeatedly rebuilt ODE models.

Model construction and integration remain with the caller. A fresh model's
effective numeric reaction bindings are copied into a private float64 vector;
only immutable generated code is reused between models. Nonparameter literals,
rate forms, stoichiometry, species order and compiler/ABI options all contribute
to the cache key. No interpreter search path changes are needed to load a cache.
"""

from copy import deepcopy
from functools import lru_cache
import ast
import hashlib
import importlib.util
import json
from numbers import Real
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

from . import solver


_COMPILE_ARGS = ["-fno-fast-math", "-ffp-contract=off"]
_LINK_ARGS = ["-fno-fast-math"]


def _parameterized_model(model):
    """Copy the model, replacing every effective numeric binding by an index."""
    # deepcopy(set) can change iteration order, which changes floating-point
    # derivative sums even when every flux is identical. The private model is
    # only prepared/checked/read after this copy; none of those operations adds
    # or removes metabolite reaction associations. Immutable snapshots preserve
    # the reference solver's exact accumulation order without sharing state.
    reaction_order = {id(met.getReactions()): tuple(met.getReactions())
                      for met in model.getMetList()}
    parameterized = deepcopy(model, reaction_order)
    values = []

    def parameter(value):
        if not isinstance(value, Real):
            return value
        # The existing generator inserts str(value) directly into expressions.
        # Parse that rendered literal rather than widening the object (float32
        # would otherwise change precision). Keep unary signs outside the
        # indexed magnitude: -2.0**2 and (-2.0)**2 have different meanings.
        try:
            expression = ast.parse(str(value), mode="eval").body
        except (SyntaxError, ValueError):
            return value
        sign = ""
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, (ast.USub, ast.UAdd)):
            sign = "-" if isinstance(expression.op, ast.USub) else "+"
            expression = expression.operand
        if isinstance(expression, ast.Constant):
            number = expression.value
        elif sys.version_info < (3, 8) and isinstance(expression, ast.Num):
            number = expression.n
        else:
            return value
        if type(number) not in (int, float):
            return value
        try:
            magnitude = float(number)
        except OverflowError:
            return value
        if isinstance(number, int) and number != magnitude:
            return value
        index = len(values)
        values.append(magnitude)
        return sign + "_odecell_params[{}]".format(index)

    for entry in parameterized.getExplicitParameters().values():
        entry.val = parameter(entry.val)
    # These are the bindings getFinalRate actually uses, including direct
    # reaction.addParameter calls and globals propagated to a reaction. The
    # optimization registry can contain older values and is not authoritative.
    for reaction in parameterized.getRxnList():
        for name, value in reaction.getKeysVals().items():
            if isinstance(value, Real):
                reaction.addParameter(name, parameter(value), verbose=0)
    return parameterized, np.asarray(values, dtype=np.double)


def _build_identity(source):
    try:
        import Cython
    except ImportError as error:
        raise ImportError("The optional Cython ODE backend requires Cython and a C compiler.") from error
    metadata = {
        "cache_format": 1,
        "python": sys.version,
        "abi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "cython": Cython.__version__,
        "compiler": sysconfig.get_config_var("CC"),
        "compiler_environment": {name: os.environ.get(name) for name in
                                 ("CC", "CXX", "CFLAGS", "CPPFLAGS", "LDFLAGS", "LDSHARED")},
        "compile_args": _COMPILE_ARGS,
        "link_args": _LINK_ARGS,
    }
    encoded = json.dumps(metadata, sort_keys=True) + "\n" + source
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return "_odecell_" + digest, metadata


def _compile_extension(source, cache, module_name, metadata):
    """Build privately, then atomically publish a complete extension file."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    destination = cache / (module_name + suffix)
    log_path = cache / (module_name + ".build.log")
    with tempfile.TemporaryDirectory(prefix="build_", dir=str(cache)) as temporary:
        temporary = Path(temporary)
        (temporary / (module_name + ".pyx")).write_text(source)
        setup = (
            "from setuptools import setup, Extension\n"
            "from Cython.Build import cythonize\n"
            "import numpy\n"
            "extension = Extension({name!r}, [{pyx!r}], include_dirs=[numpy.get_include()], "
            "extra_compile_args={compile_args!r}, extra_link_args={link_args!r})\n"
            "setup(ext_modules=cythonize([extension], compiler_directives="
            "{{'language_level': 3, 'boundscheck': False, 'wraparound': False}}))\n"
        ).format(name=module_name, pyx=module_name + ".pyx",
                 compile_args=_COMPILE_ARGS, link_args=_LINK_ARGS)
        (temporary / "setup.py").write_text(setup)
        completed = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(temporary), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        log_path.write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError("Cython ODE compilation failed; see {}".format(log_path))
        extension = temporary / (module_name + suffix)
        if not extension.is_file():
            raise RuntimeError("Cython ODE compiler did not produce {}; see {}".format(extension.name, log_path))
        # Keep source/options beside the binary for reproducibility. Concurrent
        # simulations build in separate temporary directories; only a complete
        # shared object is ever visible at the final cache path.
        (cache / (module_name + ".pyx")).write_text(source)
        (cache / (module_name + ".json")).write_text(json.dumps(metadata, indent=2, sort_keys=True))
        os.replace(str(extension), str(destination))
    return destination


@lru_cache(maxsize=8)
def _load_cached_class(source, cache_directory, identity):
    module_name, metadata_json = identity
    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    extension = cache / (module_name + sysconfig.get_config_var("EXT_SUFFIX"))
    if not extension.is_file():
        _compile_extension(source, cache, module_name, json.loads(metadata_json))
    # A unique module name for each complete source/ABI key avoids extension
    # reload behavior and avoids process-global sys.path mutation.
    spec = importlib.util.spec_from_file_location(module_name, str(extension))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.stepClass


def buildSolver(model, cache_dir=None):
    """Build a fresh parameterized callable, compiling its structure only once.

    Requires Cython, NumPy headers and a Unix C compiler on the first call for a
    new source/ABI key. Each result owns its parameter vector. Callbacks accept
    float64 NumPy state arrays, as supplied by SciPy and the normal flux saver.
    """
    parameterized, values = _parameterized_model(model)
    generator = solver.ModelSolver(parameterized)
    result = generator.buildCall(odeint=False, useJac=False, cythonBuild=True,
                                 functor=True, sourceOnly=True, typedParameters=True)
    if result == 1 or generator.cythonSource is None:
        raise ValueError("Cannot compile an invalid metabolic model")
    source = generator.cythonSource
    module_name, metadata = _build_identity(source)
    if cache_dir is None:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        cache_dir = base / "odecell" / "cython"
    identity = (module_name, json.dumps(metadata, sort_keys=True))
    functor = _load_cached_class(source, str(Path(cache_dir).expanduser().resolve()), identity)
    return functor(values)
