# ticket_processing

Turn OSC help-desk tickets into a curated set of question/answer pairs for
evaluating hpcGPT. Answers are the staff's original
replies, unmodified. The LLM is used only to *classify and score* candidates.

```
ticket CSV  ->  tickets_combined.json  ->  HPC-AgentBench  ->  canonical.jsonl
   ->  qa_pairs.json  ->  LLM judge  ->  qa_pairs_judged.jsonl  ->  top-N final set
```

## Repo layout

| Path | What it is |
|---|---|
| `scripts/` | Pipeline steps (CSV conversion, qa extraction, judging, selection, checks) |
| `slurm/` | Batch scripts that start a vLLM server on a GPU node and run a stage |
| `hpc_agentbench/` | Copy of [JinwenPan/HPC-AgentBench](https://github.com/JinwenPan/HPC-AgentBench) |

Ticket data and all outputs are git-ignored.

## Setup

```bash
git clone git@github.com:osc-hpcgpt/ticket_processing.git
cd ticket_processing

module load python/3.10            # HPC-AgentBench needs Python >= 3.10 (the default 3.9 fails)
python -m venv venv_server
source venv_server/bin/activate
pip install -r requirements.txt
```

The slurm scripts expect the venv at `<repo>/venv_server`.

## Running the pipeline

All commands run on Ascend from the repo root unless stated. 

**1. Convert the CSV to HPC-AgentBench's input format**

```bash
python scripts/csv_to_tickets.py path/to/cases.csv hpc_agentbench/tickets_combined.json
```

Splits each ticket's comment thread into customer/staff turns (staff are the
authors who appear across many tickets), strips auto-close boilerplate, and
writes the JSON the pipeline reads. 

**2. Generate canonical records - slurm**

```bash
cd hpc_agentbench 
mkdir -p logs
sbatch ../slurm/run_pipeline.sh
```

Starts a vLLM server, runs HPC-AgentBench's classify / generate / validate /
repair stages, and writes `hpc_agentbench/pipeline_outputs/canonical.jsonl`.

**3. Extract candidate QA pairs**

```bash
cd ..   # back to repo root
python scripts/extract_pairs.py hpc_agentbench/pipeline_outputs/canonical.jsonl qa_pairs.json
```

Keeps valid records whose reference reply is real, with question >= 10 words
and answer >= 30 words.

**4. Judge the pairs - slurm**

```bash
mkdir -p logs
sbatch slurm/judge.sh
```

An LLM scores each pair and tags category, customer-facing, HPC-related and
clear-QA. It never edits text. Output: `qa_pairs_judged.jsonl`. Category is 
recorded to obtain more variety in top tickets

**5. Select the final set**

```bash
python scripts/select_final.py qa_pairs_judged.jsonl \
  hpc_agentbench/pipeline_outputs/canonical.jsonl qa_pairs_final_250.json --n 250
python scripts/check_final.py qa_pairs_final_250.json
```

Applies the judge filters, prefers grounded + validated records, backfills
from the rest. Options: `--max-per-category`, `--dup-threshold`. `check_final.py` counts leftover
boilerplate and prints 5 random pairs to manually check.

**6. Map the pairs back to the original tickets**

```bash
python scripts/map_to_tickets.py qa_pairs_final_250.json path/to/cases.csv top250_tickets
```

Writes `top250_tickets.csv` (original rows) and `.json` (rows + Q/A).
