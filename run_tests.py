#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py — runs the full unittest harness for the three core
tabs (Prediction / Optimizer / Flight Envelope) and prints a summary.

Usage:
    python run_tests.py            # run everything
    python -m unittest discover -s tests -v   # equivalent, stdlib-only
"""
import sys
import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
