"""Chapman 2022 (PRApplied 18, 044027) d=2 finite-key formulas.

Cross-checks the closed forms against direct sums and pins the physical
behaviour (limits, monotonicity, death threshold) rather than magic numbers.
"""
import math
import unittest

import keyrate_ent as ke


class TestSourceModel(unittest.TestCase):
    def test_pair_prob_normalises(self):
        # (A3) sums to 1 over n (tail negligible far beyond N_MAX)
        gamma = 0.05
        total = sum(ke.pair_prob(n, gamma) for n in range(200))
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_coincidence_closed_form_matches_sum(self):
        # (A6) closed form == sum_n P_n * Y_n  (A5), summed far past N_MAX
        gamma, ea, eb, xa, xb = 0.0225, 0.3, 0.24, 1e-5, 1e-5
        direct = sum(ke.pair_prob(n, gamma) * ke.yield_n(n, ea, eb, xa, xb)
                     for n in range(200))
        closed = ke.coincidence_prob(gamma, ea, eb, xa, xb)
        self.assertAlmostEqual(closed, direct, delta=abs(direct) * 1e-9)

    def test_yield_zero_pairs_is_background_product(self):
        # Y_0 = xi_A * xi_B (sheet §3)
        self.assertAlmostEqual(ke.yield_n(0, 0.3, 0.24, 1e-5, 2e-5), 2e-10)


class TestErrorModel(unittest.TestCase):
    def test_background_errors_closed_form_matches_binomial_sum(self):
        # (A7): the code may use 1-(1-eta)^n for the inner binomial sums;
        # verify against the explicit sums from the paper.
        gamma, ea, eb, xa, xb = 0.05, 0.3, 0.24, 1e-4, 2e-4
        direct = 0.0
        for n in range(ke.N_MAX + 1):
            pn = ke.pair_prob(n, gamma)
            t1 = sum(math.comb(n, k) * ea**k * (1 - ea)**(n - k)
                     for k in range(1, n + 1)) * xb * (1 - eb)**n
            t2 = sum(math.comb(n, l) * eb**l * (1 - eb)**(n - l)
                     for l in range(1, n + 1)) * xa * (1 - ea)**n
            t3 = xa * xb * (1 - ea)**n * (1 - eb)**n
            direct += pn * (t1 + t2 + t3)
        direct *= 0.5  # e0 = 1/2 for d=2
        self.assertAlmostEqual(ke._background_errors(gamma, ea, eb, xa, xb),
                               direct, delta=direct * 1e-9)

    def test_q2_limit_is_intrinsic_error(self):
        # xi -> 0, mu tiny: only single pairs, fully detected -> Q ~ e2
        q = ke.q2_est(gamma=1e-4, eta_a=0.3, eta_b=0.24,
                      xi_a=0.0, xi_b=0.0, e2=0.01)
        self.assertAlmostEqual(q, 0.01, delta=0.002)

    def test_q2_grows_with_mu(self):
        # more pairs per pulse -> more multi-pair random bits -> higher QBER
        lo = ke.q2_est(0.005, 0.3, 0.24, 1e-5, 1e-5, 0.01)
        hi = ke.q2_est(0.1, 0.3, 0.24, 1e-5, 1e-5, 0.01)
        self.assertGreater(hi, lo)

    def test_q2_capped_at_half(self):
        # pathological params must not exceed the random-guess ceiling
        q = ke.q2_est(0.5, 1e-6, 1e-6, 0.5, 0.5, 0.3)
        self.assertLessEqual(q, 0.5)

    def test_multi_pair_errors_zero_when_bob_dark(self):
        # Physical pin for the (A12) eta_A/eta_B correction: if Bob detects
        # nothing (eta_b=0) there can be no coincidence, so no multi-pair
        # error either. The corrected form (eta_b in the second sum) gives
        # exactly 0 here.
        self.assertEqual(ke._multi_pair_errors(0.05, 0.3, 0.0), 0.0)

        # Prove the pin is not vacuous: the published (buggy) form, which
        # uses eta_a in BOTH sums and so never reads eta_b at all, breaks
        # this invariant and returns a positive number for the same inputs.
        def buggy_multi_pair_errors(gamma, eta_a, eta_b):
            total = 0.0
            for n in range(2, ke.N_MAX + 1):
                pn = ke.pair_prob(n, gamma)
                pa = 1.0 - (1.0 - eta_a) ** n
                total += pn * (pa * pa - (eta_a ** n) * (eta_a ** n))
            return 0.5 * total

        self.assertGreater(buggy_multi_pair_errors(0.05, 0.3, 0.0), 0.0)

    def test_phi_errors_n1_dominated_limit(self):
        # (A11) sanity check: for gamma -> 0, single-pair events (n=1)
        # dominate (P_n ~ gamma^n for n>=2 is negligible), and the n=1 term
        # reduces to P_1 * eta_a * eta_b * e2 (same_mode = e2^1 * 2/2 = e2).
        gamma, ea, eb, e2 = 1e-6, 0.3, 0.24, 0.01
        expected = ke.pair_prob(1, gamma) * ea * eb * e2
        actual = ke._phi_errors(gamma, ea, eb, e2)
        self.assertAlmostEqual(actual, expected, delta=expected * 1e-4)


class TestFiniteKey(unittest.TestCase):
    M = 10_000_000

    def test_key_positive_at_low_qber(self):
        ell = ke.secret_key_length_ent(self.M, 0.01)
        self.assertGreater(ell, 0)
        # rough efficiency window: ~0.7 expected, generous bounds
        self.assertGreater(ell / self.M, 0.5)
        self.assertLess(ell / self.M, 0.85)

    def test_key_dead_above_threshold(self):
        # asymptotic death: 1 = 2.12*h2(Q) -> Q ~ 11%; finite-key slightly lower
        self.assertEqual(ke.secret_key_length_ent(self.M, 0.12), 0)

    def test_key_alive_below_threshold(self):
        self.assertGreater(ke.secret_key_length_ent(self.M, 0.09), 0)

    def test_never_exceeds_raw_block(self):
        ell = ke.secret_key_length_ent(self.M, 0.0)
        self.assertLessEqual(ell, self.M)

    def test_small_block_penalised(self):
        big = ke.secret_key_length_ent(10_000_000, 0.02) / 10_000_000
        small = ke.secret_key_length_ent(100_000, 0.02) / 100_000
        self.assertLess(small, big)

    def test_scan_beats_fixed_point(self):
        # the r/beta scan must do at least as well as the fixed choice
        # (r=0.9, beta=eps_sec/8) evaluated by hand
        m, q, eps = 1_000_000, 0.02, 1e-10
        n = int(m * 0.9)
        k = m - n
        beta = eps / 8
        delta = math.sqrt((n + k) / (n * k) * (k + 1) / k * math.log(1 / beta))
        n_ext = n * (1 - ke.h2(q + delta)) - 1.12 * n * ke.h2(q) \
            - math.log2(2 / 1e-10)
        fixed = max(0, math.floor(n_ext + 4 * math.log2(beta) - 2))
        self.assertGreaterEqual(ke.secret_key_length_ent(m, q), fixed)

    def test_zero_block_returns_zero(self):
        self.assertEqual(ke.secret_key_length_ent(0, 0.01), 0)


if __name__ == "__main__":
    unittest.main()
