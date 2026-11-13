export SCIOWL_BASE_URL=https://api.deepseek.com
export SCIOWL_MODEL=deepseek-flash
export SCIOWL_API_KEY=sk-933088bcd3694acca9b8accae302071b

current_time="$(date +%Y%m%d-%H%M%S)"
ws="$PWD/runs/${SCIOWL_MODEL}-mace-${current_time}"
registry="$PWD/registry_mace.json"

# ouput_log="${ws}/run_output.json"

# Replace target.cif with your actual CIF, XYZ, or extXYZ file.
# The extra Python command extracts only the structure ID from the JSON output.
sid=$(python3 -m sciowl_agent.cli \
  --workspace "$ws" import "$PWD/mace-demo/demo.xyz" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["structure_id"])')

printf 'Structure ID: %s\n' "$sid"

# Check compatibility before proceeding:
python3 -m sciowl_agent.cli \
  --workspace "$ws" --registry "$registry" \
  tool check_potential \
  "{\"structure_id\":\"$sid\",\"potential_id\":\"mace_omat\"}"

# The worker receives the potential configuration saved in each job
python3 -m sciowl_agent.cli \
  --workspace "$ws" \
  worker --loop --progress --timeout 1800 &
worker_pid=$!
echo ${worker_pid}

session="run_output-$(date +%Y%m%d-%H%M%S)"

# let DeepSeek call MACE
# --follow waits and resumes the agent; it does not execute queued jobs
python3 -m sciowl_agent.cli \
  --workspace "$ws" \
  --registry "$registry" \
  agent \
  --base-url "$SCIOWL_BASE_URL" \
  --model "$SCIOWL_MODEL" \
  --protocol native \
  --session "$session" \
  --max-steps 20 \
  --max-jobs 2 \
  --follow \
  --wall-seconds 1800 \
  --request "Inspect structure $sid and check compatibility with potential mace_omat.
Then call evaluate_energy_forces with potential_id mace_omat and stress_policy auto.
Calculate the energy, atomic forces, and stress when available at the input geometry.
Do not perform relaxation or molecular dynamics.
Retrieve the completed result using get_result and cite its evidence ID.
If the calculation fails, report the failure accurately." \
  | tee "${ws}/${session}.json"


jid=$(python3 - "${ws}/${session}.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    state = json.load(f)

if state["status"] != "completed":
    sys.exit(f"Session is not completed: {state['status']}")
if len(state["jobs"]) != 1:
    sys.exit(f"Expected one calculation job, found {len(state['jobs'])}")

print(state["jobs"][0])
PY
)

python3 -m sciowl_agent.cli --workspace "$ws" \
  tool get_result "{\"job_id\":\"$jid\"}"

export_dir="$PWD/exports/${SCIOWL_MODEL}-mace-${current_time}"
mkdir -p "$export_dir"

python3 -m sciowl_agent.cli --workspace "$ws" \
  export-predictions \
  --job "$jid" \
  --output "$export_dir/predictions.jsonl"

kill "$worker_pid"

echo "mace demo finished..."
