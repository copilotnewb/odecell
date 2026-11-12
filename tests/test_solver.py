"""Numerical and construction regressions for the generated Python solver."""

import unittest
from unittest import mock

import numpy as np
from scipy.integrate import ode

from solver_test_support import load_solver, make_chain


modelbuilder, solver_module = load_solver()


class SolverTests(unittest.TestCase):
    def test_no_jacobian_build_does_not_evaluate_gradients(self):
        model = make_chain(modelbuilder)
        reaction = model.getReaction(0)
        solver = solver_module.ModelSolver(model)
        with mock.patch.object(reaction, "getGrad", wraps=reaction.getGrad) as gradients:
            reaction_ids = solver.buildCall(odeint=False, useJac=False)
        self.assertEqual(gradients.call_count, 0)
        self.assertEqual(reaction_ids, ["R0"])
        np.testing.assert_array_equal(solver(0.0, [2.0, 1.0]), [-1.4, 1.4])

    def test_no_jacobian_build_does_not_evaluate_auxiliary_gradients(self):
        model = make_chain(modelbuilder)
        index = model.addReaction("auxiliary", "zeroOrder")
        model.addParameter(index, "K", 4.0)
        reaction = model.getReaction(index)
        reaction.setResult("aux_value")
        solver = solver_module.ModelSolver(model)
        with mock.patch.object(reaction, "getGrad", wraps=reaction.getGrad) as gradients:
            solver.buildCall(odeint=False, useJac=False)
        self.assertEqual(gradients.call_count, 0)
        np.testing.assert_array_equal(solver.calcFlux(0.0, [2.0, 1.0]), [1.4, 4.0])

    def test_rhs_and_fluxes_match_jacobian_build(self):
        model = make_chain(modelbuilder)
        loss = model.addReaction("loss", "firstOrder")
        model.addSubstrate(loss, "Sub1", "M1")
        model.addParameter(loss, "K", 0.2)
        source = model.addReaction("source", "zeroOrder")
        model.addProduct(source, "Prod1", "M0")
        model.addParameter(source, "K", 0.05)
        model.setUnitConvFactor(2.0)
        model.setGR(0.03)
        model.getMetList()[1].setMode("odeonly")

        plain = solver_module.ModelSolver(model)
        with_jacobian = solver_module.ModelSolver(model)
        plain.buildCall(odeint=False, useJac=False)
        with_jacobian.buildCall(odeint=False, useJac=True)
        y = [2.0, 1.0]
        np.testing.assert_allclose(plain(0.0, y), [-2.76, 2.4], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(plain(0.0, y), with_jacobian(0.0, y))
        np.testing.assert_array_equal(plain.calcFlux(0.0, y), [1.4, 0.2, 0.05])
        np.testing.assert_array_equal(plain.calcFlux(0.0, y), with_jacobian.calcFlux(0.0, y))
        np.testing.assert_allclose(
            np.asarray(with_jacobian.calcJac(0.0, y)),
            [[-1.43, 0.0], [1.4, -0.4]], rtol=0, atol=1e-15,
        )

    def test_rebuilt_solver_uses_current_parameters(self):
        model = make_chain(modelbuilder)
        first = solver_module.ModelSolver(model)
        first.buildCall(odeint=False, useJac=False, noBuild=True)
        model.addParameter("R0", "K", 1.2)
        second = solver_module.ModelSolver(model)
        second.buildCall(odeint=False, useJac=False, noBuild=True)
        np.testing.assert_array_equal(first(0.0, [2.0, 1.0]), [-1.4, 1.4])
        np.testing.assert_array_equal(second(0.0, [2.0, 1.0]), [-2.4, 2.4])
        np.testing.assert_array_equal(second.calcFlux(0.0, [2.0, 1.0]), [2.4])

    def test_odeint_argument_order_is_preserved(self):
        solver = solver_module.ModelSolver(make_chain(modelbuilder))
        solver.buildCall(odeint=True, useJac=False)
        np.testing.assert_array_equal(solver([2.0, 1.0], 0.0), [-1.4, 1.4])

    def test_lsoda_trajectory_matches_analytic_chain(self):
        solver = solver_module.ModelSolver(make_chain(modelbuilder))
        solver.buildCall(odeint=False, useJac=False)
        integrator = ode(solver).set_integrator("lsoda").set_initial_value([2.0, 1.0])
        result = integrator.integrate(1.0)
        self.assertTrue(integrator.successful())
        expected_first = 2.0 * np.exp(-0.7)
        np.testing.assert_allclose(result, [expected_first, 3.0 - expected_first], rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
