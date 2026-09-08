"""The optimisation map: which policy wins at which (d_km, n_X), and by how much.

This is the deliverable the thesis title asks for -- "toi uu hoa kenh truyen va
do dai khoa" is exactly the channel axis and the block-length axis -- so the
tests here are aimed at the ways a map can look finished and be wrong: a
winner column that quietly reports the Oracle (a bound, not a policy), a margin
measured against the wrong control, a grid that skips points, and a regime
label read off the block fill time instead of off the measured coupling
between eta and the total key.
"""
import os
import shutil
import tempfile
import unittest

from evemarkov import EveParams
from hybridenv import HybridParams

import optmap


def row(policy, eta, **kw):
    """A compare() row, with only the fields the summary reads filled in."""
    out = {"policy": policy, "episodes": 1, "eta": eta, "switches": 0.0,
           "expired": 0.0, "secret_bits": 0.0, "ent_fraction": 0.0,
           "starved": 0.0, "mean_d_km": 55.0}
    out.update(kw)
    return out


def proportional(policy, eta, base=(0.60, 4.0e7)):
    """A row from a SUPPLY-limited cell: key and eta move together exactly.

    This is what n_X = 1e5 measures -- (eta/eta_pm)/(key/key_pm) is 1.0000 for
    every policy to four decimals -- so it is what the default rows here look
    like.
    """
    eta_pm, key_pm = base
    return row(policy, eta, secret_bits=key_pm * eta / eta_pm)


# A plausible cell: block-aware wins among the realisable rows, the Oracle is
# above all of them, and the relaxed control sits just above always-pm.
ROWS = [proportional("always-pm", 0.60), proportional("always-ent", 0.64),
        proportional("greedy-0.045", 0.65),
        proportional("block-aware-80", 0.70),
        proportional("always-pm-relaxed", 0.61), proportional("oracle", 0.72)]


class TestSummarisingOneCell(unittest.TestCase):
    def summary(self, rows=None, **kw):
        kw.setdefault("d_km", 55.0)
        kw.setdefault("n_x", 100_000)
        kw.setdefault("fill", (1.4, 1.1))
        return optmap.summarise_cell(rows=ROWS if rows is None else rows, **kw)

    def test_it_carries_its_own_coordinates(self):
        cell = self.summary(d_km=170.0, n_x=3_000_000)
        self.assertEqual(cell["d_km"], 170.0)
        self.assertEqual(cell["n_x"], 3_000_000)

    def test_the_winner_is_the_best_REALISABLE_row(self):
        # THE failure this map exists to avoid. The Oracle scores highest in
        # every cell by construction; a winner column that reported it would
        # be a map of a policy nobody can run.
        cell = self.summary()
        self.assertEqual(cell["winner"], "block-aware-80")
        self.assertAlmostEqual(cell["eta_winner"], 0.70)

    def test_a_bound_never_wins_even_when_it_is_the_only_thing_above_pm(self):
        rows = [row("always-pm", 0.60), row("always-pm-relaxed", 0.61),
                row("oracle", 0.90)]
        cell = self.summary(rows=rows)
        self.assertEqual(cell["winner"], "always-pm")
        self.assertAlmostEqual(cell["eta_winner"], 0.60)

    def test_the_margin_is_measured_against_always_pm(self):
        # always-pm is the control the whole study is written against; a
        # margin over the cell's mean, or over always-ent, would make cells
        # incomparable with every table already published.
        cell = self.summary()
        self.assertAlmostEqual(cell["margin"], 0.10)

    def test_the_switching_value_uses_the_RELAXED_control(self):
        # oracle - always-pm mixes the value of the relaxation into the value
        # of switching; CLAUDE.md records that being published once already.
        cell = self.summary()
        self.assertAlmostEqual(cell["switching_value"], 0.72 - 0.61)
        self.assertAlmostEqual(cell["relaxation_value"], 0.61 - 0.60)

    def test_the_headroom_left_is_what_the_winner_did_not_take(self):
        cell = self.summary()
        self.assertAlmostEqual(cell["headroom_left"], 0.72 - 0.70)

    def test_it_reports_the_fraction_of_the_headroom_the_winner_took(self):
        cell = self.summary()
        self.assertAlmostEqual(cell["headroom_taken"], 0.10 / 0.12)

    def test_a_cell_with_no_headroom_does_not_divide_by_zero(self):
        # A margin of zero over a headroom of zero. The interesting half is
        # below: a tiny epsilon denominator would pass this one too.
        rows = [row("always-pm", 0.60), row("always-pm-relaxed", 0.60),
                row("oracle", 0.60)]
        cell = self.summary(rows=rows)
        self.assertEqual(cell["headroom_taken"], 0.0)

    def test_a_nonzero_margin_over_no_headroom_is_reported_as_zero(self):
        # Physically this cannot happen -- a realisable policy above the bound
        # means the bound is broken -- but the guard has to be a guard and not
        # an epsilon in the denominator, which would turn it into 5e+10 and
        # put a nonsense number at the top of the sorted map.
        rows = [row("always-pm", 0.60), row("greedy-0.045", 0.65),
                row("always-pm-relaxed", 0.60), row("oracle", 0.60)]
        cell = self.summary(rows=rows)
        self.assertAlmostEqual(cell["margin"], 0.05)
        self.assertEqual(cell["headroom_taken"], 0.0)

    def test_it_records_the_block_fill_times_that_decide_the_regime(self):
        # n_X / ns_m in slots. This is the number that explains the cell, and
        # a map without it invites the reader to explain a flipped winner by
        # the distance when the block length did it.
        cell = self.summary(fill=(14.4, 10.8))
        self.assertAlmostEqual(cell["fill_pm"], 14.4)
        self.assertAlmostEqual(cell["fill_ent"], 10.8)

    def test_a_cell_where_eta_tracks_the_key_exactly_is_supply_limited(self):
        cell = self.summary(fill=(1.4, 1.1), t0_deadline=20)
        self.assertEqual(cell["regime"], "supply")
        self.assertLess(cell["coupling_spread"], 1e-9)

    def test_a_cell_where_more_key_buys_less_eta_is_timing_limited(self):
        # THE measured signature of the timing regime, and it is not
        # hypothetical: at n_X = 1e6 block-aware distils MORE key than any
        # other realisable policy and delivers LESS eta.
        rows = [proportional("always-pm", 0.60),
                proportional("always-ent", 0.64),
                row("block-aware-80", 0.62, secret_bits=4.6e7),
                proportional("always-pm-relaxed", 0.61),
                proportional("oracle", 0.72)]
        cell = self.summary(rows=rows, fill=(14.4, 10.8), t0_deadline=20)
        self.assertEqual(cell["regime"], "timing")
        self.assertGreater(cell["coupling_spread"], optmap.SUPPLY_SPREAD)

    def test_the_regime_is_measured_and_not_read_off_the_fill_time(self):
        # Same rows, absurd fill times: the label must not move, because the
        # fill time is a diagnostic and the coupling is the evidence.
        near = self.summary(fill=(1.0, 1.0), t0_deadline=20)
        far = self.summary(fill=(400.0, 900.0), t0_deadline=20)
        self.assertEqual(near["regime"], far["regime"])

    def test_a_cell_whose_control_delivered_nothing_has_no_regime(self):
        rows = [row("always-pm", 0.0, secret_bits=0.0),
                row("always-ent", 0.4, secret_bits=1e7),
                row("always-pm-relaxed", 0.0, secret_bits=0.0),
                row("oracle", 0.5, secret_bits=2e7)]
        cell = self.summary(rows=rows)
        self.assertIsNone(cell["coupling_spread"])
        self.assertEqual(cell["regime"], "unknown")

    def test_a_dead_cell_is_flagged_rather_than_reported_as_a_win(self):
        rows = [row("always-pm", 0.0), row("always-ent", 0.0),
                row("always-pm-relaxed", 0.0), row("oracle", 0.0)]
        self.assertEqual(self.summary(rows=rows)["regime"], "dead")
        cell = self.summary(rows=rows)
        self.assertTrue(cell["dead"])

    def test_a_live_cell_is_not_flagged_dead(self):
        self.assertFalse(self.summary()["dead"])

    def test_it_refuses_a_cell_with_no_control_row(self):
        # Without always-pm there is nothing to measure a margin against, and
        # silently reporting 0.0 would read as "switching bought nothing".
        # The message must name the CELL: substituting a synthetic zero
        # control still raises further down in coupling_spread, so a bare
        # assertRaises would pass while the summary quietly invented one.
        with self.assertRaises(KeyError) as ctx:
            self.summary(d_km=170.0, rows=[row("greedy-0.045", 0.7)])
        self.assertIn("d=170.0", str(ctx.exception))


class TestTheGridIsVisitedInFull(unittest.TestCase):
    def test_every_point_of_the_product_grid_becomes_a_cell(self):
        seen = []

        def fake_cell(d_km, n_x, **kw):
            seen.append((d_km, n_x))
            return optmap.summarise_cell(d_km=d_km, n_x=n_x, rows=ROWS,
                                         fill=(1.0, 1.0))

        cells = optmap.sweep_map(d_grid=(55.0, 170.0),
                                 n_x_grid=(100_000, 1_000_000),
                                 seeds=(1,), cell_runner=fake_cell)
        self.assertEqual(len(cells), 4)
        self.assertEqual(set(seen), {(55.0, 100_000), (55.0, 1_000_000),
                                     (170.0, 100_000), (170.0, 1_000_000)})

    def test_the_cells_come_back_in_grid_order(self):
        cells = optmap.sweep_map(
            d_grid=(55.0, 170.0), n_x_grid=(100_000, 1_000_000), seeds=(1,),
            cell_runner=lambda d_km, n_x, **kw: optmap.summarise_cell(
                d_km=d_km, n_x=n_x, rows=ROWS, fill=(1.0, 1.0)))
        self.assertEqual([(c["d_km"], c["n_x"]) for c in cells],
                         [(55.0, 100_000), (55.0, 1_000_000),
                          (170.0, 100_000), (170.0, 1_000_000)])


class TestRenderingTheMap(unittest.TestCase):
    def cells(self):
        return optmap.sweep_map(
            d_grid=(55.0, 170.0), n_x_grid=(100_000, 1_000_000), seeds=(1,),
            cell_runner=lambda d_km, n_x, **kw: optmap.summarise_cell(
                d_km=d_km, n_x=n_x, rows=ROWS, fill=(1.0, 1.0)))

    def test_the_table_names_every_winner(self):
        text = optmap.format_map(self.cells())
        self.assertIn("block-aware-80", text)

    def test_the_table_shows_both_axes(self):
        text = optmap.format_map(self.cells())
        self.assertIn("170", text)
        self.assertIn("1e+06", text)

    def test_the_csv_holds_one_line_per_policy_per_cell(self):
        cells = self.cells()
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "map.csv")
            optmap.write_rows_csv(cells, path)
            with open(path) as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        finally:
            shutil.rmtree(tmp)
        # 4 cells x 6 policy rows, plus the header.
        self.assertEqual(len(lines), 1 + 4 * len(ROWS))
        self.assertIn("d_km", lines[0])
        self.assertIn("n_x", lines[0])
        self.assertIn("policy", lines[0])


class TestRunningARealCell(unittest.TestCase):
    """One genuine cell, tiny, so the wiring to eval_hybrid.compare is real."""

    def test_it_scores_a_real_environment_and_reports_the_fill_times(self):
        cell = optmap.run_cell(
            d_km=55.0, n_x=100_000, seeds=(99_000_000,),
            params=HybridParams(horizon=60, lam=2.5, b_max=1_000_000,
                                ent_preset="ent_advanced"),
            eve=EveParams(), thresholds=(0.045,), reserves=(0.0,),
            horizons=(20,), shapes=((0.0, 0),))
        self.assertEqual(cell["d_km"], 55.0)
        self.assertGreater(cell["fill_pm"], 0.0)
        self.assertGreater(cell["fill_ent"], 0.0)
        self.assertIn("always-pm", [r["policy"] for r in cell["rows"]])
        self.assertNotIn(cell["winner"], optmap.BOUND_NAMES)

    def test_the_block_length_reaches_the_environment(self):
        # A map whose n_X never left the argument list would draw a flat map
        # and look perfectly healthy.
        small = optmap.run_cell(
            d_km=55.0, n_x=100_000, seeds=(99_000_000,),
            params=HybridParams(horizon=60, lam=2.5, b_max=1_000_000,
                                ent_preset="ent_advanced"),
            eve=EveParams(), thresholds=(0.045,), reserves=(0.0,),
            horizons=(20,), shapes=((0.0, 0),))
        big = optmap.run_cell(
            d_km=55.0, n_x=1_000_000, seeds=(99_000_000,),
            params=HybridParams(horizon=60, lam=2.5, b_max=1_000_000,
                                ent_preset="ent_advanced"),
            eve=EveParams(), thresholds=(0.045,), reserves=(0.0,),
            horizons=(20,), shapes=((0.0, 0),))
        self.assertAlmostEqual(big["fill_pm"] / small["fill_pm"], 10.0,
                               places=6)


class TestJammingCanBeTurnedOff(unittest.TestCase):
    def test_the_quiet_adversary_has_every_level_at_zero(self):
        # The title is "under a jamming attack", so the map's contrast case
        # has to be a real no-jamming run rather than a lower HEAVY level.
        quiet = optmap.quiet_eve()
        self.assertEqual(set(quiet.f_levels), {0.0})

    def test_it_keeps_the_chain_shape_of_the_adversary_it_replaces(self):
        # Only the levels are zeroed: the dwell structure stays, so the two
        # maps differ in jamming alone and not in the Markov chain as well.
        loud = EveParams(transition=((0.9, 0.1, 0.0),
                                     (0.1, 0.8, 0.1),
                                     (0.0, 0.2, 0.8)))
        self.assertEqual(optmap.quiet_eve(loud).transition, loud.transition)


if __name__ == "__main__":
    unittest.main()
