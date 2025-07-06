import os
import subprocess

def run_evaluations():
    """Run evaluations for multiple checkpoints"""

    exp_ids = ["SafeDiffCon"]  # List of experiment IDs to evaluate
    gpu_id = 0
    # checkpoints = list(range(1,2))
    checkpoints = [190]
    
    for exp_id in exp_ids:
        for ckpt in checkpoints:
            # Build command line arguments
            cmd = [
                "python", "eval.py",
                "--checkpoint", str(ckpt),
                "--exp_id", str(exp_id),
                "--gpu_id", str(gpu_id)
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