"""
Unified Audit Test Suite Runner
===============================
Executes:
1. Environment check
2. Pipeline smoke test
3. Focused unit test suite:
   - SICT tests (cases A-E, differentiability)
   - Label coding boundary tests (u=0, 640, out-of-frame, inverse consistency)
   - Structural loss edge-case tests (absent, valid, one, two, alternating, curvature)
   - Metric calculation tests (TP/FP/FN/TN, 8px tolerance, RMSE)
   - Model tests (parameter count 2,205,242, [B, 18, 101], CPU & CUDA forward)
   - Deterministic reproducibility test
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.env_check import check_environment
from scripts.smoke_test import run_smoke_test


def run_all_audit_tests():
    print("=" * 75)
    print("STEP 2 - COMPREHENSIVE RESEARCH IMPLEMENTATION AUDIT")
    print("=" * 75)

    # 1. Environment Verification
    print("\n[PHASE 1/3] ENVIRONMENT COMPATIBILITY AUDIT")
    print("-" * 50)
    env = check_environment()
    for k, v in env.items():
        print(f"  {k:22s}: {v}")
    assert env["torch_version"] != "Missing", "PyTorch missing!"
    assert env["torchvision_version"] != "Missing", "Torchvision missing!"
    print("  -> Environment compatibility verified.")

    # 2. Pipeline Smoke Test
    print("\n[PHASE 2/3] PIPELINE SMOKE TEST")
    print("-" * 50)
    smoke_res = run_smoke_test()
    assert smoke_res["status"] == "PASSED", "Smoke test failed!"

    # 3. Comprehensive Unit Test Suite
    print("\n[PHASE 3/3] FOCUSED UNIT TEST SUITE")
    print("-" * 50)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 75)
    print("AUDIT TEST SUMMARY")
    print("=" * 75)
    print(f"Total Unit Tests Run: {result.testsRun}")
    print(f"Passed:               {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed:               {len(result.failures)}")
    print(f"Errors:               {len(result.errors)}")
    print(f"Skipped:              {len(result.skipped)}")
    
    success = result.wasSuccessful()
    print(f"OVERALL AUDIT STATUS: {'PASSED' if success else 'FAILED'}")
    print("=" * 75)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": success,
        "env": env,
        "smoke": smoke_res,
    }


if __name__ == "__main__":
    res = run_all_audit_tests()
    sys.exit(0 if res["success"] else 1)
