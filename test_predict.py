"""
Minimal Test Script - Save Test Parameters
===========================================
Simply saves hardcoded test parameters to file.

Usage:
    test_predictor.exe input_file.txt output_file.txt

Arguments:
    input_file.txt  - Input file (ignored for test)
    output_file.txt - Output file for parameters

Exit Codes:
    0 - Success
    1 - Error (missing arguments or file write error)
"""

import sys


def save_test_params(input_path, output_path):
    """Save hardcoded test parameters.

    Args:
        input_path: Input file path (ignored for test)
        output_path: Output file path for parameters

    Returns:
        int: Exit code (0 = success, 1 = error)
    """

    try:
        # Test parameters (from screenshot)
        params = {
            'Dmax1': 0.01305,
            'D01': 0.0017,
            'L1': 5800,
            'Rp1': 3500,
            'D02': 0.004845,
            'L2': 4000,
            'Rp2': -500,
            'Dmin': 0.0001
        }

        # Write to output file
        with open(output_path, 'w') as f:
            f.write("# Test Parameters\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

        # Success
        return 0

    except Exception as e:
        # Error - write to log file (no console available)
        try:
            with open('test_predictor_error.log', 'w') as f:
                f.write(f"Error: {str(e)}\n")
        except:
            pass
        return 1


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) != 3:
        # Missing arguments
        try:
            with open('test_predictor_error.log', 'w') as f:
                f.write(f"Error: Missing arguments\n")
                f.write(f"Usage: test_predictor.exe input_file.txt output_file.txt\n")
                f.write(f"Got {len(sys.argv)-1} arguments\n")
        except:
            pass
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    exit_code = save_test_params(input_file, output_file)
    sys.exit(exit_code)
