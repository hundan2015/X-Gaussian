"""Simple lazy GPU scheduler for running shell-command tasks."""

import os
import sys
import subprocess
import argparse
import torch
import time
from typing import List, Tuple, Optional, Any
from collections import deque
import glob
import shlex


class GPUTaskManager:
    """Run named shell tasks, assigning GPUs only when free."""

    def __init__(
        self, tasks_per_gpu: int = 2, verbose: bool = True, logs_dir: str = "./logs"
    ):
        """Configure slots per GPU, verbosity, and log folder."""
        self.tasks_per_gpu = tasks_per_gpu
        self.verbose = verbose
        actual_gpus = torch.cuda.device_count()
        # Use at least 1 slot so CPU-only can still queue/run tasks
        self.num_gpus = actual_gpus if actual_gpus > 0 else 1
        self.processes = []
        # Store tasks as (name, command_string) tuples
        self.task_configs: List[Tuple[str, str]] = []
        self.logs_dir = logs_dir

        if self.verbose:
            detected = torch.cuda.device_count()
            print(f"[GPUTaskManager] Detected {detected} GPU(s)")
            if torch.cuda.is_available() and detected > 0:
                for i in range(detected):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"  GPU {i}: {gpu_name}")
            else:
                print("  CUDA not available; tasks will run without GPU binding")

    def add_task(self, name: str, command: str):
        """Register a task (name, command string)."""

        self.task_configs.append((name, command))

        if self.verbose:
            print(f"[Task Added] {name} | Command: {command}")

    # build_command removed; tasks provide full commands directly

    def run_task(
        self, name: str, cmd: str, task_id: int, gpu_id: int | None
    ) -> Optional[subprocess.Popen[Any]]:
        """Start one task on a chosen GPU and return the process."""
        # Set CUDA_VISIBLE_DEVICES to the specific GPU
        env = os.environ.copy()
        if torch.cuda.is_available() and gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if self.verbose:
            gpu_msg = gpu_id if gpu_id is not None else "N/A"
            printable_cmd = cmd
            print(f"\n[Task {task_id}] Starting on GPU {gpu_msg}")
            print(f"  Command: {printable_cmd}")
            print(f"  Name: {name}")

        try:
            # Start the training process
            os.makedirs(self.logs_dir, exist_ok=True)
            readable_ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            output_log = os.path.join(self.logs_dir, f"log_{readable_ts}_{name}.txt")
            os.makedirs(os.path.dirname(output_log), exist_ok=True)

            with open(output_log, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_file,
                    stderr=log_file,
                    text=True,
                    bufsize=1,
                    shell=True,
                )
            return process
        except Exception as e:
            print(f"[ERROR] Failed to start task {task_id}: {e}")
            return None

    def run_all(self):
        """Run all queued tasks, launching when GPUs have capacity."""
        if not self.task_configs:
            print("[WARNING] No tasks configured!")
            return

        max_tasks = self.tasks_per_gpu

        print(f"\n[Starting Training] Total tasks: {len(self.task_configs)}")
        print(f"  Max tasks per GPU: {max_tasks}")
        print(f"  Total GPUs: {self.num_gpus}")
        print("=" * 60)

        # Track running processes
        running_processes = {}  # {process: (task_id, name, gpu_id)}
        task_queue = deque(
            enumerate(self.task_configs)
        )  # queue of (task_id, (name, cmd))
        gpu_task_count = {i: 0 for i in range(self.num_gpus)}  # Track tasks per GPU

        # Helper to find an available GPU id (first-fit)
        def _acquire_gpu():
            if self.num_gpus == 0:
                return None
            for gid in range(self.num_gpus):
                if gpu_task_count[gid] < max_tasks:
                    return gid
            return None

        # Monitor and manage running processes with a simple loop
        try:
            while running_processes or len(task_queue) > 0:
                # Launch tasks while there is capacity
                launched = False
                while len(task_queue) > 0:
                    next_gpu_id = _acquire_gpu()
                    if next_gpu_id is None:
                        break
                    next_task_id, (next_name, next_cmd) = task_queue.popleft()
                    proc = self.run_task(next_name, next_cmd, next_task_id, next_gpu_id)
                    if proc:
                        running_processes[proc] = (next_task_id, next_name, next_gpu_id)
                        gpu_task_count[next_gpu_id] += 1
                        launched = True

                # Check for finished processes
                finished = []
                for proc, (tid, tname, gid) in list(running_processes.items()):
                    if proc.poll() is None:
                        continue
                    return_code = proc.returncode
                    status = (
                        "✓ SUCCESS"
                        if return_code == 0
                        else f"✗ FAILED (code: {return_code})"
                    )
                    print(f"\n[Task {tid}] {status} | Name: {tname} | GPU {gid}")
                    if gid is not None:
                        gpu_task_count[gid] -= 1
                    finished.append(proc)
                for proc in finished:
                    del running_processes[proc]

                # Print status
                if running_processes:
                    status_str = " | ".join(
                        [f"GPU{i}: {gpu_task_count[i]}" for i in range(self.num_gpus)]
                    )
                    print(
                        f"\r[Status] Running tasks: {len(running_processes)} | {status_str}",
                        end="",
                        flush=True,
                    )

                if not launched:
                    time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Terminating all processes...")
            for proc in list(running_processes.keys()):
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            print("[INTERRUPT] All processes terminated")

        print("\n" + "=" * 60)
        print("[Training Complete]")


def config_parser():
    """Build CLI parser for demo usage."""
    parser = argparse.ArgumentParser(description="Multi-GPU training script")

    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=1,
        help="Number of tasks to run per GPU (default: 1)",
    )
    parser.add_argument(
        "--shutdown", action="store_true", help="Shutdown the system after completion"
    )
    parser.add_argument("--output", type=str, default="./logs/", help="Default output")
    parser.add_argument("--input", type=str, default="./data/", help="Default input")

    return parser


def example_usage():
    """Demo: create tasks from data files and run them."""
    parser = config_parser()
    args = parser.parse_args()
    # Place logs under the provided output folder
    manager = GPUTaskManager(
        tasks_per_gpu=args.tasks_per_gpu,
        verbose=True,
        logs_dir=os.path.join(args.output, "logs"),
    )

    # Example 1: Add individual tasks
    # Adjust these paths based on your actual data and configs
    pickle_files = glob.glob(f"{args.input}/*.pickle")
    for i, pickle_file in enumerate(pickle_files):
        pickle_file_name = os.path.basename(pickle_file).split(".")[0]
        # SAX-NeRF
        output_dir = os.path.join(args.output, pickle_file_name)
        os.makedirs(output_dir, exist_ok=True)
        # Align command with launch.json: use train.py and its arguments
        cmd_args = [
            sys.executable,
            "train.py",
            "--source_path",
            pickle_file,
            "--iterations",
            "20",
            "--model_path",
            f"{output_dir}",
            "--eval",
        ]
        cmd = shlex.join(cmd_args)
        manager.add_task(f"x_gaussian_{pickle_file_name}", cmd)

    manager.run_all()

    if args.shutdown:
        print("[Shutdown] System will shutdown.")
        os.system("shutdown -h now")


if __name__ == "__main__":
    example_usage()
