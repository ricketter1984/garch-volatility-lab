"""
Unit tests for the stage_analysis package using synthetic price paths.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_analysis import Universe, scan
from stage_analysis.data import to_weekly
from stage_analysis.report import build_html
from stage_analysis.stages import analyze, classify_series


def make_path(segments, seed=0, start=50.0, noise=0.006, vol=2_000_000):
    """Build daily OHLCV from (n_days, total_pct_change) segments."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate([np.full(n, np.log1p(pct / 100.0) / n) for n, pct in segments])
    close = start * np.exp(np.cumsum(drift + rng.normal(0, noise, len(drift))))
    idx = pd.bdate_range(end="2026-10-09", periods=len(close))
    high = close * (1 + np.abs(rng.normal(0.015, 0.004, len(close))))
    low = close * (1 - np.abs(rng.normal(0.015, 0.004, len(close))))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(int(vol * 0.8), int(vol * 1.2), len(close)).astype(float)},
                        index=idx)


SHAPES = {
    "BASE": [(250, 0), (200, -45), (300, 0)],                   # decline then long flat base -> Stage 1
    "UP":   [(300, 0), (450, 150)],                             # steady advance -> Stage 2
    "DOWN": [(300, 0), (450, -60)],                             # steady decline -> Stage 4
    "TOP":  [(150, 0), (300, 120), (300, 0)],                   # advance then flat -> Stage 3
    "BRK":  [(250, 0), (200, -40), (270, 0), (30, 15)],         # base then breakout -> 2A
}


def synthetic_prices():
    p = {t: make_path(s, seed=i) for i, (t, s) in enumerate(SHAPES.items())}
    p["SPY"] = make_path([(750, 20)], seed=99)
    p["XLK"] = make_path([(750, 40)], seed=98)
    p["SMH"] = make_path([(750, 80)], seed=97)
    return p


class TestStages(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_prices()
        self.bench_w = to_weekly(self.prices["SPY"])["Close"]

    def stage_of(self, t):
        d = self.prices[t]
        return analyze(to_weekly(d), self.bench_w, d)

    def test_basic_stages(self):
        self.assertEqual(self.stage_of("BASE")["stage"], 1)
        self.assertEqual(self.stage_of("UP")["stage"], 2)
        self.assertEqual(self.stage_of("DOWN")["stage"], 4)
        self.assertEqual(self.stage_of("TOP")["stage"], 3)

    def test_breakout_is_scored(self):
        rec = self.stage_of("BRK")
        self.assertIn(rec["stage"], (1, 2))
        self.assertFalse(np.isnan(rec["transition_score"]))
        self.assertGreaterEqual(rec["transition_score"], 0)
        self.assertLessEqual(rec["transition_score"], 100)

    def test_classify_series_has_all_columns(self):
        cls = classify_series(to_weekly(self.prices["UP"]))
        self.assertListEqual(list(cls.columns), ["close", "ma30", "ma_slope", "stage"])
        self.assertTrue(set(cls["stage"].dropna().unique()) <= {1, 2, 3, 4})


class TestScan(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_prices()
        u = Universe(sector_etfs={"XLK": "Technology"}, theme_etfs={"SMH": "AI Semis"})
        for t in SHAPES:
            u.add(t, "Technology", ["Test Theme", "Shape " + t])
        self.res = scan(self.prices, u)

    def test_scan_tables(self):
        self.assertEqual(len(self.res.stocks), len(SHAPES))
        for col in ("stage", "substage", "transition_score", "adr20_pct", "variant1", "variant2", "RS_3M"):
            self.assertIn(col, self.res.stocks.columns)
        self.assertIn("Rotation", self.res.theme_baskets.columns)
        self.assertIn("Test Theme", self.res.theme_baskets.index)
        self.assertAlmostEqual(self.res.theme_baskets.loc["Test Theme", "%S1"]
                               + self.res.theme_baskets.loc["Test Theme", "%S2"]
                               + self.res.theme_baskets.loc["Test Theme", "%S3"]
                               + self.res.theme_baskets.loc["Test Theme", "%S4"], 100.0)
        self.assertListEqual(list(self.res.sector_etfs.index), ["XLK"])

    def test_views(self):
        self.assertIn("UP", self.res.view("2").index)
        self.assertIn("DOWN", self.res.view("4").index)
        self.assertNotIn("UP", self.res.view("4").index)
        self.assertEqual(len(self.res.view("all")), len(SHAPES))

    def test_html(self):
        html = build_html(self.res)
        self.assertIn("Stage Analysis Scanner", html)
        self.assertIn('"BASE"', html)
        self.assertNotIn("NaN", html)


if __name__ == "__main__":
    unittest.main()
