import os
import subprocess

def run_evaluations():
    """Run evaluations for multiple checkpoints"""

    exp_ids = ["turbo-1"]  # List of experiment IDs to evaluate
    gpu_id = 5
    checkpoints = list(range(171, 172))
    # checkpoints = [100]
    use_max_safety = [True]  # If xxx-repeat, must be True

    for exp_id, use_max_safety in zip(exp_ids, use_max_safety):  # Iterate over each experiment ID and corresponding safety setting
        for ckpt in checkpoints:
            print(f"\nStarting evaluation for checkpoint {ckpt} in experiment {exp_id}")
            
            # Build command line arguments
            cmd = [
                "python", "eval.py",
                "--checkpoint", str(ckpt),
                "--exp_id", str(exp_id),
                "--gpu_id", str(gpu_id),
                "--use_max_safety", str(use_max_safety)
            ]
            
            # Run evaluation
            try:
                subprocess.run(cmd, check=True)
                print(f"Checkpoint {ckpt} evaluation completed for {exp_id}")
            except subprocess.CalledProcessError as e:
                print(f"Checkpoint {ckpt} evaluation failed for {exp_id}: {e}")
                continue

if __name__ == "__main__":
    run_evaluations()