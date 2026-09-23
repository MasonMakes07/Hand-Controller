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

    def test_tracking_loss_releases_inputs_and_resumes_outside_box(self):
        self.controller.update((0.6, 0.5), {"fist"}, 2)
        self.assertEqual(self.controller.update(None, {"fist"}, 3)[0], set())
        self.assertEqual(self.controller.center, (0.5, 0.5))
        self.assertEqual(self.controller.update((0.8, 0.8), {"fist"}, 4)[0],
                         {"right", "down", "jump"})
        self.assertEqual(self.controller.update((0.5, 0.5), set(), 5)[0], set())
        self.assertEqual(self.controller.center, (0.5, 0.5))

    def test_gestures_and_movement_across_frame_with_configured_box(self):
        import config
        controller = OneHandController(deadzone=config.ONE_HAND_DEADZONE)
        controller.update((0.5, 0.5), set(), 0)
        controller.update((0.5, 0.5), set(), 1)
        for point, directions in [((0.05, 0.05), {"left", "up"}),
                                  ((0.95, 0.05), {"right", "up"}),
                                  ((0.05, 0.95), {"left", "down"}),
                                  ((0.95, 0.95), {"right", "down"})]:
            with self.subTest(point=point):
                self.assertEqual(controller.update(point, {"thumbs_up"}, 2)[0],
                                 directions | {"interact"})
                self.assertEqual(controller.center, (0.5, 0.5))

    def test_tracking_loss_clears_direction_latch(self):
        self.controller.update((0.7, 0.5), set(), 2)
        self.controller.update(None, set(), 3)
        self.assertEqual(self.controller.update((0.57, 0.5), set(), 4)[0], set())

    def test_loss_during_initial_calibration_restarts_hold(self):
        controller = OneHandController()
        controller.update((0.5, 0.5), set(), 0)
        controller.update(None, set(), 0.9)
        controller.update((0.5, 0.5), {"fist"}, 1)
        self.assertIsNone(controller.center)
        self.assertEqual(controller.update((0.5, 0.5), {"fist"}, 2)[0], set())
        self.assertEqual(controller.center, (0.5, 0.5))

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
