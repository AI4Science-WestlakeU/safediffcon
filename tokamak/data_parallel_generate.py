import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_simulation(seed):
    """Run the KSTAR data generator with the specified seed."""
    try:
        command = ["python", "kstar_data_generator_random_target.py", "--seed", str(seed)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Seed {seed} failed with error:\n{result.stderr}")
        else:
            print(f"Seed {seed} completed successfully.")
    except Exception as e:
        print(f"Seed {seed} failed with exception: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run KSTAR Data Generator with multiple seeds.")
    parser.add_argument("--num_seeds", type=int, default=50000, help="Number of random seeds to use.")
    parser.add_argument("--max_threads", type=int, default=120, help="Maximum number of threads to use.")

    args = parser.parse_args()

    # Generate seed list
    seeds = list(range(args.num_seeds))

    # Run simulations using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        executor.map(run_simulation, seeds)

    print("All simulations have been submitted.")
