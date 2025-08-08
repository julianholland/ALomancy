#!/usr/bin/env python3
"""
Test runner script for alomancy package.

This script provides convenient ways to run different types of tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        if capture_output:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run alomancy tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "slow", "all"], 
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--no-cov", 
        action="store_true", 
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "test_files", 
        nargs="*", 
        help="Specific test files to run"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add coverage
    if args.coverage or (not args.no_cov and args.type in ["all", "integration"]):
        cmd.extend([
            "--cov=alomancy",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    
    # Add test type markers
    if args.type == "unit":
        cmd.extend(["-m", "unit or not (integration or slow or requires_external)"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "slow":
        cmd.extend(["-m", "slow"])
    elif args.type == "all":
        pass  # Run all tests
    
    # Add specific test files
    if args.test_files:
        cmd.extend(args.test_files)
    else:
        cmd.append("tests/")
    
    # Run the tests
    run_command(cmd)


if __name__ == "__main__":
    main()
