import glob
import os
import shlex

import submitit




executor = submitit.AutoExecutor(folder="log_submitit")

ONE_DAY_IN_MIN = 1 * 24 * 60
executor.update_parameters(
    name="GRPO_Math_Training",
    timeout_min=int(1.0 * ONE_DAY_IN_MIN),
    gpus_per_node=1,
    cpus_per_task=12,
    slurm_array_parallelism=24,
    mem_gb=120,
    slurm_account=os.environ["SLURM_ACCT_NAME"],
    slurm_partition=os.environ.get("SLURM_PARTITION_NAME"),
    slurm_qos=os.environ["SLURM_QOS"],
)

# Find all checkpoint directories
checkpoint_base = "/checkpoint/memorization/saeedm/jxm/mr/checkpoints/verl-math-training-2-11-07"
checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_base, "*")))
run_names = [os.path.basename(path) for path in checkpoint_paths if os.path.isdir(path)]

print(f"Found {len(run_names)} checkpoints:")
for run_name in run_names:
    print(f"  - {run_name}")
print()

command_str = """HDFS_HOME={checkpoint_base} RUN_NAME="{run_name}" VLLM_ATTENTION_BACKEND=FLASH_ATTN  bash eval_math_nodes.sh --run_name {run_name} --init_model Qwen2.5-Math-7B --template qwen-boxed --tp_size 2 --add_step_0 true --temperature 1.0 --top_p 0.95 --max_tokens 16000 --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math --n_sampling 1"""

jobs = []

idx = 0
with executor.batch():
    for run_name in run_names:
        result_command = command_str.format(run_name=run_name).strip()
        print(f"Job {idx+1}/{len(run_names)}: {run_name}")
        print(f">> Command: {result_command}")
        function = submitit.helpers.CommandFunction(
            shlex.split(result_command),
        )
        print()
        job = executor.submit(function)
        jobs.append(job)
        idx += 1

if jobs:
    print(f"*** SUBMITIT: Successfully submitted {len(jobs)} jobs: {jobs[0]._job_id} .. {jobs[-1]._job_id} ***")
else:
    print("*** SUBMITIT: No jobs submitted (no checkpoints found) ***")