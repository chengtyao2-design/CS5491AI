
import os
import sys
import pathlib

# Ensure the root directory is in sys.path so 'implementation' can be imported
ROOT_DIR = pathlib.Path(__file__).parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the main entry point from implementation.funsearch
try:
    from implementation import funsearch
    from implementation import config as config_lib
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)

if __name__ == '__main__':
    print(f"Running FunSearch from: {ROOT_DIR}")
    
    # Path to the specification file
    SPEC_FILE = ROOT_DIR / 'implementation' / 'specification_nonsymmetric_admissible_set.txt'
    
    if SPEC_FILE.exists():
        with open(SPEC_FILE, 'r') as f:
            specification_content = f.read()
        
        # Define test inputs (n, w)
        # Example inputs: dimension n=12, weight w=7
        test_inputs = [(8, 4), (12, 7)] 

        # Load default configuration
        default_config = config_lib.Config()
        
        print(f"Specification: {SPEC_FILE.name}")
        print(f"Test Inputs: {test_inputs}")
        
        # Launch the main function
        funsearch.main(specification_content, test_inputs, default_config)
    else:
        print(f"Error: Specification file not found at {SPEC_FILE}")
