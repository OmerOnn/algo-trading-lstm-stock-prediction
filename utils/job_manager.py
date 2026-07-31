import os
import sys
import re
import subprocess
from typing import List, Dict, Set, Optional


# --------------------------------------------------------------------
# Helpers to query SLURM
# --------------------------------------------------------------------

def get_partition_nodes(partition: str) -> Set[str]:
    """
    Return the set of hostnames that belong to the given SLURM partition.
    """
    cmd = f'sinfo -h --Node --partition={partition} --Format="NodeHost"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running sinfo for partition {partition}: {result.stderr}")

    nodes: Set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # sinfo -h with NodeHost usually prints a hostname in the first column
        hostname = line.split()[0]
        nodes.add(hostname)
    return nodes


def get_top_n_nodes_with_max_cpus_and_mem(
    n: int = 10,
    cpu_partition: str = "cpu",
    gpu_partition: str = "gpu",
    reserve_cpus_per_node: int = 4,
    mem_fraction: float = 0.9,
) -> List[Dict[str, int]]:
    """
    Returns top N *CPU-only* nodes (no GPUs) sorted by idle CPUs.

    Each returned dict has:
        {
            'hostname': <str>,
            'cpus': <usable idle CPUs>,
            'mem': <usable free memory in GB>
        }

    Logic:
      1) cpu_nodes = nodes in cpu_partition
      2) gpu_nodes = nodes in gpu_partition
      3) cpu_only_nodes = cpu_nodes - gpu_nodes   (strictly no-GPU machines)
      4) Query cpu_partition for CPU/mem details and keep only cpu_only_nodes
    """
    # 1) Get node sets per partition
    cpu_nodes = get_partition_nodes(cpu_partition)
    gpu_nodes = get_partition_nodes(gpu_partition)

    # 2) CPU-only nodes: in cpu partition but not in gpu partition
    cpu_only_nodes = cpu_nodes - gpu_nodes

    if not cpu_only_nodes:
        raise RuntimeError("No CPU-only nodes found (cpu partition minus gpu partition is empty).")

    # 3) Query cpu partition for CPU/mem details
    cmd = (
        f'sinfo -r --Node --partition={cpu_partition} '
        '--exact --Format="NodeHost,CPUsState,FreeMem"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error running sinfo command: {result.stderr}")

    nodes: List[Dict[str, int]] = []
    seen_nodes: Set[str] = set()

    pattern = re.compile(
        r'(?P<hostname>\S+)\s+'
        r'(?P<cpus_alloc>\d+)/(?P<cpus_idle>\d+)/(?P<cpus_other>\d+)/(?P<cpus_total>\d+)\s+'
        r'(?P<free_mem>\d+)'
    )

    for line in result.stdout.splitlines():
        m = pattern.match(line)
        if not m:
            continue

        info = m.groupdict()
        hostname = info["hostname"]

        # Skip header and duplicates
        if hostname in seen_nodes or hostname == "HOSTNAMES":
            continue
        seen_nodes.add(hostname)

        # Keep only strict CPU-only nodes (no overlap with gpu partition)
        if hostname not in cpu_only_nodes:
            continue

        cpus_idle = int(info["cpus_idle"])
        free_mem_mb = int(info["free_mem"])

        # Safety margin: leave a few CPUs free on each node
        usable_cpus = max(cpus_idle - reserve_cpus_per_node, 0)
        if usable_cpus <= 0:
            continue

        # Safety margin on memory: use only a fraction of free mem
        mem_gb = int((free_mem_mb / 1024.0) * mem_fraction)
        if mem_gb <= 0:
            continue

        nodes.append({
            "hostname": hostname,
            "cpus": usable_cpus,
            "mem": mem_gb,
        })

    # Sort by usable CPUs, descending
    nodes_sorted = sorted(nodes, key=lambda x: x["cpus"], reverse=True)
    return nodes_sorted[:n]


# --------------------------------------------------------------------
# Job Manager
# --------------------------------------------------------------------

class JobManager:
    def __init__(
        self,
        main_path: str,
        num_jobs: int,
        parent_path: str = os.getcwd(),
        env_path: str = sys.executable,
        log_data_path: str = os.path.join(os.getcwd(), "main_logs"),
        partition: str = "cpu",
        max_cpus_per_job: int = 64,
        total_cpu_limit: int = 2600 - 128,
        fixed_mem_gb: Optional[int] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        """
        main_path: path to the Python script to run.
        num_jobs:  number of jobs to submit.
        partition: SLURM partition to use (default: 'cpu').
        max_cpus_per_job: upper bound on CPUs per job to avoid huge jobs.
        total_cpu_limit: global CPU limit across all jobs (cluster-wide cap).
        """
        self.main_path = main_path
        self.args = ""
        self.parent_path = parent_path
        self.env_path = env_path
        self.log_data_path = log_data_path
        self.partition = partition
        self.max_cpus_per_job = max_cpus_per_job
        self.fixed_mem_gb = fixed_mem_gb
        self.env_vars = env_vars or {}

        # CPU-only nodes: in cpu partition but not in gpu partition
        self.free_resources = get_top_n_nodes_with_max_cpus_and_mem(
            n=num_jobs,
            cpu_partition=self.partition,
            gpu_partition="gpu",
        )

        self.num_jobs = num_jobs
        self.job_scripts: List[str] = []
        self.job_ids: List[str] = []

        self.limit_cpu = total_cpu_limit
        self.current_limit_cpu = total_cpu_limit

        os.makedirs(self.log_data_path, exist_ok=True)

    def create_job_script(self, job_index: int) -> str:
        """
        Create a single job script file and return its path.
        Returns None if no CPUs left to assign.
        """
        node_info = self.free_resources[job_index]
        mem = self.fixed_mem_gb if self.fixed_mem_gb is not None else int(node_info["mem"] * 0.97)
        cpus_from_node = node_info["cpus"]

        print(f"[JobManager] Job {job_index}: node={node_info['hostname']} cpus_available={cpus_from_node} mem_available={mem}G", flush=True)
        print(f"[JobManager] command: {self.env_path} {self.main_path} {self.args}", flush=True)
        
        # Cap CPUs per job and by remaining global limit
        cpus_job = min(cpus_from_node, self.max_cpus_per_job, self.current_limit_cpu)

        if cpus_job <= 0:
            return None

        self.current_limit_cpu -= cpus_job

        job_script_name = os.path.join(self.log_data_path, f"job_script_{job_index}.sh")

        with open(job_script_name, "w") as script_file:
            script_file.write("#!/bin/bash\n")
            script_file.write(f"#SBATCH --job-name=multi_job_{job_index}\n")
            script_file.write(f"#SBATCH --cpus-per-task={cpus_job}\n")
            script_file.write(f"#SBATCH --mem={mem}G\n")
            script_file.write(
                f"#SBATCH --output={os.path.join(self.log_data_path, f'%j_job_output_{job_index}.txt')}\n"
            )
            script_file.write(f"#SBATCH --partition={self.partition}\n")
            script_file.write(f"export PYTHONPATH={self.parent_path}:$PYTHONPATH\n")
            for key, value in self.env_vars.items():
                script_file.write(f"export {key}={value}\n")
            script_file.write(f"{self.env_path} {self.main_path} {self.args}\n")
        return job_script_name

    def create_jobs(self):
        """
        Create scripts for up to num_jobs jobs and submit them.
        """
        for job_index in range(self.num_jobs):
            job_script_name = self.create_job_script(job_index)
            if job_script_name is not None:
                self.job_scripts.append(job_script_name)

        self.submit_jobs()
        self.cleanup()

    def submit_jobs(self):
        """
        Submit all created job scripts with sbatch.
        """
        for script in self.job_scripts:
            result = subprocess.run(
                ["sbatch", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if result.returncode != 0:
                print(f"sbatch error for {script}:", result.stderr, file=sys.stderr)
                continue

            out = result.stdout.strip().split()
            if not out:
                print(f"Unexpected sbatch output for {script}: {result.stdout}", file=sys.stderr)
                continue

            job_id = out[-1]
            self.job_ids.append(job_id)

        print("Submitted jobs with IDs:", self.job_ids, flush=True)

    def cleanup(self):
        """
        Remove the temporary job script files.
        """
        for script in self.job_scripts:
            try:
                os.remove(script)
            except OSError:
                pass


# --------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------

if __name__ == "__main__":
    icarus_directory = os.getcwd()

    # >>> CHANGE THIS TO YOUR SCRIPT <<<
    main_path = "/home/roeeidan/icarus_framework/test.py"

    parent_path = icarus_directory
    env_path = sys.executable
    log_data_path = os.path.join(icarus_directory, "main_logs")

    num_jobs = 10

    manager = JobManager(
        main_path=main_path,
        num_jobs=num_jobs,
        parent_path=parent_path,
        env_path=env_path,
        log_data_path=log_data_path,
        partition="cpu",          # use the cpu partition
        max_cpus_per_job=64,      # cap CPUs per job
        total_cpu_limit=2600 - 128,
    )
    manager.create_jobs()
