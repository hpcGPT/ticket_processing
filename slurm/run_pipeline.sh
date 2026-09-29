#!/bin/bash -l
#SBATCH --account=PDE0050
#SBATCH --job-name=qa_pipeline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<EMAIL>

# Submit from INSIDE hpc_agentbench/ 

# Usage: sbatch ../slurm/run_pipeline.sh

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
PORT=8000

cd "$SLURM_SUBMIT_DIR" || { echo "cd failed"; exit 1; }

# process_tickets.py resolves tickets_combined.json relative to its own
# location and ignores --input with --stage all, so it must run from here.
if [ ! -f process_tickets.py ]; then
  echo "process_tickets.py not found in $PWD -- submit from hpc_agentbench/"
  exit 1
fi
if [ ! -f tickets_combined.json ]; then
  echo "tickets_combined.json missing from $PWD"
  exit 1
fi

module load vllm/0.23.0
export HF_HOME=/fs/ess/PDE0050/vllm/.cache/huggingface

echo "=== launching vllm serve on port $PORT ==="
# OSC's vllm_start/vllm_stop wrappers only exist in interactive sessions,
# so call vllm serve directly. --max-model-len is required: the model's
# 262k default needs 24 GiB of KV cache and won't fit on a 40GB A100.
vllm serve "$MODEL" --port $PORT --max-model-len 8192 \
  > "logs/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
echo "vLLM server PID: $VLLM_PID"

echo "=== waiting for server to become ready ==="
READY=0
for i in $(seq 1 40); do
  if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "Server is up after $((i * 15)) seconds"
    READY=1
    break
  fi
  # Stop early if the server process already died.
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM exited early -- see logs/vllm_server_${SLURM_JOB_ID}.log"
    exit 1
  fi
  echo "waiting... ($i/40)"
  sleep 15
done

if [ "$READY" -ne 1 ]; then
  echo "Server never became ready, aborting"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

module load python/3.10
source ../venv_server/bin/activate

echo "=== running pipeline ==="
# -u keeps progress visible in the log; without it Python block-buffers
# stdout when redirected to a file and the log looks frozen for hours.
python -u process_tickets.py \
  --stage all \
  --backend server \
  --server-base-url "http://localhost:$PORT/v1" \
  --batch-size 8 \
  --model "$MODEL" \
  --output pipeline_outputs

echo "=== stopping vLLM server ==="
kill $VLLM_PID
