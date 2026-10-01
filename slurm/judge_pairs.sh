#!/bin/bash -l
#SBATCH --account=PDE0050
#SBATCH --job-name=qa_judge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/judge_%j.log
#SBATCH --error=logs/judge_%j.err

# Run from the repo root: sbatch slurm/judge.sh
cd "$SLURM_SUBMIT_DIR" || { echo "cd failed"; exit 1; }
mkdir -p logs

MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
PORT=8000
INPUT=qa_pairs.json
OUTPUT=qa_pairs_judged.jsonl

module load vllm/0.23.0
export HF_HOME=/fs/ess/PDE0050/vllm/.cache/huggingface

echo "=== launching vllm serve on port $PORT ==="

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
  echo "waiting... ($i/40)"
  sleep 15
done

if [ "$READY" -ne 1 ]; then
  echo "Server never became ready, aborting"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

module load python/3.10
source venv_server/bin/activate

echo "=== judging QA pairs ==="

python -u scripts/judge_pairs.py "$INPUT" "$OUTPUT" \
  --server-base-url "http://localhost:$PORT/v1" \
  --model "$MODEL"

echo "=== stopping vLLM server ==="
kill $VLLM_PID
