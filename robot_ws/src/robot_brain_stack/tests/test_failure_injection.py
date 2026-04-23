import unittest

from robot_brain_stack.failure_injection import should_inject_failure


class FailureInjectionTests(unittest.TestCase):
    def test_disabled_injection_never_fires(self):
        for _ in range(20):
            self.assertFalse(should_inject_failure(False, 0.9))

    def test_zero_rate_never_fires(self):
        for _ in range(20):
            self.assertFalse(should_inject_failure(True, 0.0))

    def test_full_rate_always_fires(self):
        for _ in range(20):
            self.assertTrue(should_inject_failure(True, 1.0))


if __name__ == '__main__':
    unittest.main()
