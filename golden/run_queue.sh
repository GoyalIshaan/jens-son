#!/bin/sh
# Runs Beam jobs one after another, retrying while the account's CPU quota is full.
# usage: sh run_queue.sh "<args 1>" "<args 2>" ...   (each arg string is passed to beam_exp2.py)
B=${BEAM:-$HOME/.local/share/uv/tools/beam-client/bin/python}
L=${LOGDIR:-results/logs}
mkdir -p "$L"
for job in "$@"; do
  name=$(echo "$job" | tr ' +' '__')
  while true; do
    $B beam_exp2.py $job > "$L/$name.log" 2>&1
    if grep -q "quota exceeded" "$L/$name.log"; then
      echo "$(date +%H:%M) $job: quota full, retrying in 3 min"; sleep 180
    else
      echo "$(date +%H:%M) $job: finished"; break
    fi
  done
done
