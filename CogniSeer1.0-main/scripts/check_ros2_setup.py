#!/usr/bin/env python3
"""
Diagnostic tool to verify ROS2 setup and dependencies.

Usage:
    python scripts/check_ros2_setup.py [--verbose] [--fix]
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class ROS2DiagnosticError(Exception):
    """ROS2 diagnostic check failed."""
    pass


def run_command(cmd: list, check: bool = False) -> tuple[int, str, str]:
    """
    Run a shell command and return exit code, stdout, stderr.
    
    Args:
        cmd: Command and arguments
        check: If True, raise on non-zero exit
        
    Returns:
        (exit_code, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise ROS2DiagnosticError(
                f"Command failed: {' '.join(cmd)}\nstderr: {result.stderr}"
            )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError as e:
        raise ROS2DiagnosticError(f"Command not found: {cmd[0]}") from e


def check_ros2_command() -> bool:
    """Check if 'ros2' command is available."""
    logger.info("Checking for 'ros2' command...")
    code, stdout, stderr = run_command(["which", "ros2"])
    if code == 0:
        logger.info(f"  ✓ ros2 found at: {stdout}")
        return True
    else:
        logger.warning("  ✗ ros2 command not found in PATH")
        return False


def check_ros_distro() -> str | None:
    """Check which ROS2 distro is active."""
    logger.info("Checking ROS_DISTRO environment variable...")
    code, distro, _ = run_command(["bash", "-c", "echo $ROS_DISTRO"])
    
    if distro:
        logger.info(f"  ✓ ROS_DISTRO={distro}")
        return distro
    else:
        logger.warning("  ✗ ROS_DISTRO not set (ROS2 environment may not be sourced)")
        logger.info("    Run: source /opt/ros/humble/setup.bash")
        return None


def check_rclpy_import() -> bool:
    """Check if rclpy can be imported."""
    logger.info("Checking for rclpy Python module...")
    try:
        import rclpy
        logger.info(f"  ✓ rclpy available from: {rclpy.__file__}")
        return True
    except ImportError as e:
        logger.warning(f"  ✗ Cannot import rclpy: {e}")
        return False


def check_ros2_version() -> str | None:
    """Get ROS2 version."""
    logger.info("Checking ROS2 version...")
    code, stdout, _ = run_command(["ros2", "--version"])
    
    if code == 0:
        logger.info(f"  ✓ {stdout}")
        return stdout
    else:
        logger.warning("  ✗ Cannot determine ROS2 version")
        return None


def check_project_installed() -> bool:
    """Check if cogniseer project is installed."""
    logger.info("Checking if project is installed...")
    try:
        import train_brain
        logger.info("  ✓ Project modules importable")
        return True
    except ImportError as e:
        logger.warning(f"  ✗ Project not installed: {e}")
        logger.info("    Run: pip install -e .")
        return False


def check_torch_import() -> bool:
    """Check if PyTorch is available."""
    logger.info("Checking for PyTorch...")
    try:
        import torch
        logger.info(f"  ✓ torch {torch.__version__} available")
        return True
    except ImportError:
        logger.warning("  ✗ PyTorch not installed")
        return False


def check_venv_active() -> bool:
    """Check if a virtual environment is active."""
    logger.info("Checking if virtual environment is active...")
    code, venv_path, _ = run_command(["bash", "-c", "echo $VIRTUAL_ENV"])
    
    if venv_path:
        logger.info(f"  ✓ Virtual environment active: {venv_path}")
        return True
    else:
        logger.warning("  ⚠️  No virtual environment active (not required, but recommended)")
        return False


def diagnose_all(verbose: bool = False) -> dict:
    """Run all diagnostic checks."""
    results = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        "checks": {},
    }

    checks = [
        ("venv_active", check_venv_active, "Python virtual environment is active"),
        ("torch", check_torch_import, "PyTorch is installed"),
        ("project", check_project_installed, "Project is installed"),
        ("ros2_cmd", check_ros2_command, "ros2 command is available"),
        ("ros_distro", check_ros_distro, "ROS_DISTRO is set"),
        ("ros2_version", check_ros2_version, "ROS2 version can be determined"),
        ("rclpy", check_rclpy_import, "rclpy module is importable"),
    ]

    for check_name, check_fn, description in checks:
        try:
            result = check_fn()
            results["checks"][check_name] = {
                "passed": bool(result),
                "description": description,
                "details": str(result) if result else None,
            }
        except ROS2DiagnosticError as e:
            logger.error(f"  Error during {check_name}: {e}")
            results["checks"][check_name] = {
                "passed": False,
                "description": description,
                "error": str(e),
            }

    # Determine overall status
    all_passed = all(c["passed"] for c in results["checks"].values())
    results["status"] = "✅ OK" if all_passed else "⚠️  ISSUES DETECTED"

    return results


def print_summary(results: dict):
    """Print diagnostic summary."""
    print("\n" + "=" * 70)
    print("ROS2 DEPENDENCY DIAGNOSTIC")
    print("=" * 70)

    for check_name, check_result in results["checks"].items():
        status = "✓" if check_result["passed"] else "✗"
        desc = check_result.get("description", check_name)
        print(f"{status} {desc}")

        if check_result.get("error"):
            print(f"  Error: {check_result['error']}")

    print("\n" + "=" * 70)
    print(f"Status: {results['status']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )

    try:
        results = diagnose_all(verbose=args.verbose)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_summary(results)

        # Guidance
        if not results["checks"]["ros2_cmd"]["passed"]:
            print("\nℹ️  ROS2 not found. If you're only doing training/inference:")
            print("   - ROS2 is OPTIONAL")
            print("   - You can safely ignore this")
            print("   - ROS2 is required only for deploying on ROS2 robots")
            print("\nTo install ROS2, see: docs/ROS2_SETUP.md")

        if not results["checks"]["rclpy"]["passed"]:
            print("\nℹ️  rclpy not available. If you're only doing training/inference:")
            print("   - This is fine—ROS2 is optional for this project")
            print("   - Run training: python train_brain.py")
            print("   - Run inference: python quantize_brain.py")
            print("\nTo use ROS2 components:")
            print("   1. Install ROS2: https://docs.ros.org/en/humble/Installation.html")
            print("   2. Source ROS2 environment: source /opt/ros/humble/setup.bash")
            print("   3. See docs/ROS2_SETUP.md for detailed setup")

        # Return non-zero if critical checks failed
        critical = ["torch", "project"]
        critical_failed = any(
            not results["checks"][c]["passed"] for c in critical if c in results["checks"]
        )
        return 1 if critical_failed else 0

    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=args.log_level == "DEBUG")
        return 1


if __name__ == "__main__":
    sys.exit(main())
