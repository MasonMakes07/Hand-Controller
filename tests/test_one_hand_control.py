import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handcontroller"))
from one_hand_control import OneHandController


class OneHandTests(unittest.TestCase):
    def setUp(self):
        self.controller = OneHandController()
        self.controller.update((0.5, 0.5), set(), 0)
        self.controller.update((0.5, 0.5), set(), 1)

    def test_directions_diagonals_and_rest(self):
        for point, expected in [((0.4, 0.4), {"left", "up"}),
                                ((0.6, 0.6), {"right", "down"}),
                                ((0.5, 0.5), set())]:
            self.assertEqual(self.controller.update(point, set(), 2)[0], expected)

    def test_hysteresis(self):
        self.controller.update((0.6, 0.5), set(), 2)
        self.assertEqual(self.controller.update((0.57, 0.5), set(), 3)[0], {"right"})
        self.assertEqual(self.controller.update((0.54, 0.5), set(), 4)[0], set())

    def test_actions_while_moving(self):
        self.assertEqual(self.controller.update((0.6, 0.5), {"fist"}, 2)[0], {"right", "jump"})
        self.assertEqual(self.controller.update((0.5, 0.5), {"thumbs_up"}, 3)[0], {"interact"})

    def test_tracking_loss_requires_new_calibration(self):
        self.controller.update((0.6, 0.5), {"fist"}, 2)
        self.assertEqual(self.controller.update(None, {"fist"}, 3)[0], set())
        self.assertIsNone(self.controller.center)
        self.assertEqual(self.controller.update((0.8, 0.8), {"fist"}, 4)[0], set())
        self.assertEqual(self.controller.update((0.8, 0.8), {"fist"}, 5)[0], set())
        self.assertEqual(self.controller.center, (0.8, 0.8))

    def test_pause_resume_and_recenter(self):
        self.controller.toggle_pause()
        self.assertEqual(self.controller.update((0.7, 0.7), {"fist"}, 2)[0], set())
        self.controller.toggle_pause()
        self.assertIsNone(self.controller.center)
        self.controller.update((0.5, 0.5), set(), 3)
        self.controller.update((0.5, 0.5), set(), 4)
        self.controller.reset()
        self.assertIsNone(self.controller.center)

    def test_movement_restarts_calibration(self):
        controller = OneHandController()
        controller.update((0.5, 0.5), set(), 0)
        controller.update((0.7, 0.5), set(), 0.9)
        controller.update((0.7, 0.5), set(), 1.1)
        self.assertIsNone(controller.center)
        controller.update((0.7, 0.5), set(), 2)
        self.assertEqual(controller.center, (0.7, 0.5))


if __name__ == "__main__":
    unittest.main()
