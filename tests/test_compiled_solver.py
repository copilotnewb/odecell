"""Real Cython/compiler checks for the optional cached metabolic callback."""

import copy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest import mock

import numpy as np

from solver_test_support import load_solver, make_chain


HAS_CYTHON = importlib.util.find_spec("Cython") is not None
COMPILER = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
HAS_COMPILER = shutil.which(shlex.split(COMPILER)[0]) is not None


@unittest.skipUnless(HAS_CYTHON and HAS_COMPILER, "Cython and the configured C compiler are required (try CC=gcc)")
class CompiledSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder, cls.solver_module, cls.compiled = load_solver(include_compiled=True)
        cls.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2])
        cls.cache = Path(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_numeric_updates_reuse_class_and_keep_old_parameters_isolated(self):
        model = make_chain(self.builder)
        original_bindings = copy.deepcopy(model.getReaction(0).getKeysVals())
        first = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertEqual(model.getReaction(0).getKeysVals(), original_bindings)
        model.addParameter("R0", "K", 1.2)
        with mock.patch.object(self.compiled, "_compile_extension", side_effect=AssertionError("recompiled unchanged structure")):
            second = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertIs(type(first), type(second))
        self.assertIsNot(first.params, second.params)
        y = np.array([2.0, 1.0])
        np.testing.assert_array_equal(first(0.0, y), [-1.4, 1.4])
        np.testing.assert_array_equal(second(0.0, y), [-2.4, 2.4])
        np.testing.assert_array_equal(second.calcFlux(0.0, y), [2.4])

    def test_new_structure_builds_a_different_class(self):
        first = self.compiled.buildSolver(make_chain(self.builder, 2), cache_dir=self.cache)
        second = self.compiled.buildSolver(make_chain(self.builder, 3), cache_dir=self.cache)
        self.assertIsNot(type(first), type(second))
        np.testing.assert_allclose(second(0.0, np.array([2.0, 1.0, 0.0])), [-1.4, 0.7, 0.7], rtol=0, atol=1e-15)

    def test_growth_and_unit_conversion_literals_are_in_the_cache_key(self):
        model = make_chain(self.builder)
        model.setGR(0.1)
        model.setUnitConvFactor(2.0)
        model.getMetList()[1].setMode("odeonly")
        first = self.compiled.buildSolver(model, cache_dir=self.cache)
        np.testing.assert_array_equal(first(0.0, np.array([2.0, 1.0])), [-3.0, 2.8])
        model.setGR(0.2)
        second = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertIsNot(type(first), type(second))
        np.testing.assert_allclose(second(0.0, np.array([2.0, 1.0])), [-3.2, 2.8], rtol=0, atol=1e-15)

    def test_preserves_reference_order_for_cancelling_reaction_sums(self):
        model = self.builder.MetabolicModel()
        model.setVerbosity(0)
        model.addMetabolite("S", initVal=2.0)
        rates = {0: 1e16, 1: 0.0, 5: 1.0, 13: -1e16}
        for index in range(14):
            reaction = model.addReaction("R" + str(index), "zeroOrder")
            model.addParameter(reaction, "K", rates.get(index, 0.0))
            if index in rates:
                model.addProduct(reaction, "Prod1", "S")
        reference = self.solver_module.ModelSolver(model)
        reference.buildCall(odeint=False, useJac=False)
        compiled = self.compiled.buildSolver(model, cache_dir=self.cache)
        y = np.array([2.0])
        # Reconstructing a set during deepcopy can change its iteration order:
        # moving +1 before -1e16 loses the small contribution entirely.
        np.testing.assert_array_equal(compiled(0.0, y), reference(0.0, y))

    def test_explicit_and_direct_reaction_parameters_are_refreshed(self):
        model = make_chain(self.builder)
        model.addParameter("Explicit", "scale", 2.0, parName="scale")
        model.getReaction(0).addParameter("K", "scale")
        first = self.compiled.buildSolver(model, cache_dir=self.cache)
        model.getExplicitParameters()["scale"].val = 3.0
        second = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertIs(type(first), type(second))
        np.testing.assert_array_equal(first.calcFlux(0.0, np.array([2.0, 1.0])), [4.0])
        np.testing.assert_array_equal(second.calcFlux(0.0, np.array([2.0, 1.0])), [6.0])
        # The effective binding can be changed directly, without updating the
        # model's optimization registry. Generated Python uses this value too.
        model.getReaction(0).addParameter("K", 4.0)
        direct = self.compiled.buildSolver(model, cache_dir=self.cache)
        np.testing.assert_array_equal(direct.calcFlux(0.0, np.array([2.0, 1.0])), [8.0])

    def test_signed_literal_precedence_and_negative_zero_are_preserved(self):
        model = self.builder.MetabolicModel()
        model.setVerbosity(0)
        model.addMetabolite("S", initVal=1.0)
        model.addRateForm("squared", self.builder.RateForm("$K**2 * $Sub1"))
        reaction = model.addReaction("square", "squared")
        model.addSubstrate(reaction, "Sub1", "S")
        model.addParameter(reaction, "K", -2.0)
        negative = self.compiled.buildSolver(model, cache_dir=self.cache)
        y = np.ones(1)
        # The existing textual generator emits -2.0**2, whose unary minus is
        # outside exponentiation. A signed value inside an array changes it.
        np.testing.assert_array_equal(negative.calcFlux(0.0, y), [-4.0])
        model.addParameter(reaction, "K", -0.0)
        zero = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertTrue(np.signbit(zero.calcFlux(0.0, y)[0]))
        model.addParameter(reaction, "K", 2.0)
        positive = self.compiled.buildSolver(model, cache_dir=self.cache)
        self.assertIsNot(type(negative), type(positive))
        np.testing.assert_array_equal(positive.calcFlux(0.0, y), [4.0])

    def test_float32_parameter_uses_the_original_rendered_literal(self):
        model = make_chain(self.builder, rate=np.float32(0.1))
        compiled = self.compiled.buildSolver(model, cache_dir=self.cache)
        # str(np.float32(0.1)) is '0.1'; widening the original object instead
        # would introduce 0.10000000149011612 into the generated model.
        np.testing.assert_array_equal(compiled.calcFlux(0.0, np.ones(2)), [0.1])

    def test_nonliteral_numeric_expression_keeps_its_source_semantics(self):
        model = self.builder.MetabolicModel()
        model.setVerbosity(0)
        model.addMetabolite("S", initVal=1.0)
        model.addRateForm("fraction", self.builder.RateForm("1 / $K * $Sub1"))
        reaction = model.addReaction("fraction", "fraction")
        model.addSubstrate(reaction, "Sub1", "S")
        model.addParameter(reaction, "K", Fraction(1, 2))
        compiled = self.compiled.buildSolver(model, cache_dir=self.cache)
        # The original expression is 1 / 1/2 * S, not 1 / (1/2) * S.
        np.testing.assert_array_equal(compiled.calcFlux(0.0, np.ones(1)), [0.5])

    def test_disk_cache_loads_without_invoking_the_compiler(self):
        first = self.compiled.buildSolver(make_chain(self.builder), cache_dir=self.cache)
        self.compiled._load_cached_class.cache_clear()
        with mock.patch.object(self.compiled, "_compile_extension", side_effect=AssertionError("compiled an existing disk cache")):
            second = self.compiled.buildSolver(make_chain(self.builder), cache_dir=self.cache)
        np.testing.assert_array_equal(first(0.0, np.ones(2)), second(0.0, np.ones(2)))

    def test_double_time_and_noncontracted_arithmetic(self):
        model = self.builder.MetabolicModel()
        model.setVerbosity(0)
        model.addMetabolite("S", initVal=1.0)
        model.addRateForm("precision", self.builder.RateForm("$A*$Sub1+$B+t"))
        reaction = model.addReaction("precision", "precision")
        model.addSubstrate(reaction, "Sub1", "S")
        model.addParameter(reaction, "A", 1.0 + 2.0 ** -27)
        model.addParameter(reaction, "B", -1.0)
        compiled = self.compiled.buildSolver(model, cache_dir=self.cache)
        # Multiplication rounds to one before addition. A contracted FMA would
        # produce -2**-54, and a float32 time would drop the final increment.
        t = 1.0 + 2.0 ** -30
        y = np.array([1.0 - 2.0 ** -27])
        np.testing.assert_array_equal(compiled.calcFlux(t, y), [t])
        model.addParameter(reaction, "A", 1e-300)
        model.addParameter(reaction, "B", 0.0)
        small = self.compiled.buildSolver(model, cache_dir=self.cache)
        value = small.calcFlux(0.0, np.array([1e-12]))[0]
        self.assertGreater(value, 0.0)
        self.assertEqual(value, np.float64(1e-300) * np.float64(1e-12))

    def test_concurrent_processes_publish_complete_shared_cache_entries(self):
        shared = self.cache / "concurrent"
        program = (
            "import sys,json; import numpy as np; "
            "sys.path.insert(0, 'tests'); "
            "from solver_test_support import load_solver,make_chain; "
            "builder,solver,compiled=load_solver(include_compiled=True); "
            "callback=compiled.buildSolver(make_chain(builder,4),cache_dir=sys.argv[1]); "
            "print(json.dumps(callback(0.0,np.array([1.,2.,3.,4.])).tolist()))"
        )
        root = Path(__file__).resolve().parents[1]
        processes = [subprocess.Popen(
            [sys.executable, "-c", program, str(shared)], cwd=str(root),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        ) for _ in range(2)]
        try:
            for process in processes:
                output, errors = process.communicate(timeout=90)
                self.assertEqual(process.returncode, 0, errors)
                np.testing.assert_allclose(json.loads(output), [-0.7, -0.7, -0.7, 2.1], rtol=0, atol=1e-15)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        self.assertEqual(len(list(shared.glob("*.so"))), 1)


if __name__ == "__main__":
    unittest.main()
