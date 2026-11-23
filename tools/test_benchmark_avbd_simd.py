"""Regression for incomplete event windows in paired headless cost reports."""
import unittest
from benchmark_avbd_simd import summarize_comparisons


def row(label, cost=1, events=None, unavailable=None):
    return dict(scene='volume', workers=8, repeat=0, label=label, passed=True,
                windows={'1-360': dict(meanMs=cost, p95Ms=cost), **(events or {})},
                unavailableWindows=unavailable or {})


class PairedWindows(unittest.TestCase):
    def test_whole_step_pairs(self):
        report = dict(runs=[row('baseline', 2), row('candidate', 1)])
        summarize_comparisons(report, [8], 1)
        self.assertTrue(report['completeWindows'])
        self.assertEqual(report['comparisons'][0]['medianRatios']['meanMs'], .5)

    def test_missing_unload_preserves_cost_and_incompleteness(self):
        report = dict(runs=[row('baseline', 2, {'unload': dict(meanMs=2, p95Ms=3)}),
                            row('candidate', 1, unavailable={'unload': 'eventNotObserved'})])
        summarize_comparisons(report, [8], 1)
        self.assertFalse(report['completeWindows'])
        whole, unload = report['comparisons']
        self.assertEqual(whole['medianRatios']['meanMs'], .5)
        self.assertFalse(unload['complete'])
        self.assertEqual(unload['missing'][0]['reason'], 'eventNotObserved')

    def test_both_unobserved_is_still_incomplete(self):
        report = dict(runs=[row(label, unavailable={'unload': 'insufficientPostEventFrames'})
                            for label in ('baseline', 'candidate')])
        summarize_comparisons(report, [8], 1)
        self.assertFalse(report['completeWindows'])
        self.assertEqual(len(report['comparisons'][1]['missing']), 2)

    def test_interrupted_pair_is_incomplete(self):
        report = dict(runs=[row('baseline')])
        summarize_comparisons(report, [8], 1)
        self.assertFalse(report['completeWindows'])


if __name__ == '__main__':
    unittest.main()
