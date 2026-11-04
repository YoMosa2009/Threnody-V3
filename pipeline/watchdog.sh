#!/usr/bin/env bash
# Threnody watchdog (runs beside run_all.sh / run_v2.sh). Every 5 min it copies a heartbeat to Drive and, on trouble,
# writes ALERT.txt there (+ an optional ntfy push). Drive FUSE only syncs closed files, so both are written locally
# then copied.
#   setsid nohup bash pipeline/watchdog.sh > /dev/null 2>&1 &
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
L=$DRIVE/logs; M=$DRIVE/markers
idle=0; last_alert=""
while true; do
  now=$(date -u +%FT%TZ)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)
  util=${gpu##*, }
  procs=$(ps aux | grep -E '[t]rain_sft|[m]erge.py|[b]ench_colab|[l]lama-|[c]onvert_hf|[b]uild_data|[b]uild_v2|[g]en_teacher|[g]en_v2|[g]ates.py|[c]kpt_gate|[v]llm serve|[p]ip install|[c]make' | awk '{print $12,$13}' | sort -u | tr '\n' ';')
  # follow whichever pipeline log is newest
  RL=$L/run_all.log; pfx=""
  [ "$L/run_v2.log" -nt "$RL" ] 2>/dev/null && { RL=$L/run_v2.log; pfx="v2_"; }
  [ "$L/run_v2_lean.log" -nt "$RL" ] 2>/dev/null && { RL=$L/run_v2_lean.log; pfx="v2_"; }
  [ "$L/run_v3.log" -nt "$RL" ] 2>/dev/null && { RL=$L/run_v3.log; pfx="v3_"; }
  running=$(pgrep -f 'pipeline/[r]un_(all|v2|v3)' > /dev/null && echo yes || echo no)
  stage=$(grep -o 'start [a-z_0-9]*' "$RL" 2>/dev/null | tail -1 | cut -d' ' -f2)
  slog="$L/$pfx$stage.log"
  steps=$( { tr '\r' '\n' < "$slog" 2>/dev/null | grep -E "'loss'|'eval_loss'|^GATE|^\{" | tail -2; \
             tr '\r' '\n' < "$slog" 2>/dev/null | grep -E "it/s\]|s/it\]" | tail -1; } )
  stale=$(( ( $(date +%s) - $(stat -c %Y "$slog" 2>/dev/null || date +%s) ) / 60 ))
  {
    echo "time=$now pipeline=$(basename $RL .log) running=$running stage=$stage gpu_mem_MiB,util%=$gpu log_age_min=$stale"
    echo "procs=$procs"
    echo "markers: $(ls $M 2>/dev/null | tr '\n' ' ')"
    echo "--- $(basename $RL)"; tail -5 "$RL" 2>/dev/null
    echo "--- $pfx$stage.log"; echo "$steps"
    [ -f "$L/v2_gate.log" ] && { echo "--- v2_gate.log"; grep '^GATE' "$L/v2_gate.log" | tail -3; }
    [ -f "$L/v3_gate.log" ] && { echo "--- v3_gate.log"; grep '^GATE' "$L/v3_gate.log" | tail -3; }
  } > /tmp/hb.txt
  cp /tmp/hb.txt "$DRIVE/heartbeat.txt"

  alert=""
  if tr '\r' '\n' < "$slog" 2>/dev/null | tail -40 | grep -qE 'Traceback|Error:|CUDA out of memory|Killed'; then
    alert="ERROR in $pfx$stage.log"
  fi
  f=$(grep -h '^!!! STAGE' "$RL" 2>/dev/null | tail -1); [ -n "$f" ] && alert="${alert:+$alert; }$f"
  g=$(grep -hE '^GATE (STOP|ERROR)|^GATE FAILED' "$L/v2_gate.log" "$L/v2_select.log" "$L/v3_gate.log" "$L/v3_select.log" 2>/dev/null | tail -1)
  [ -n "$g" ] && alert="${alert:+$alert; }$g"
  if [ "$util" -lt 5 ] 2>/dev/null && [ -z "$procs" ]; then idle=$((idle + 1)); else idle=0; fi
  done_re='ALL STAGES FINISHED|V2 FINISHED|V3 FINISHED'
  [ $idle -ge 2 ] && ! grep -qE "$done_re" "$RL" && alert="${alert:+$alert; }GPU idle with no pipeline process"
  [ "$stage" = sft ] && [ $stale -ge 45 ] && alert="${alert:+$alert; }sft log silent ${stale}m"
  grep -qE "$done_re" "$RL" 2>/dev/null && alert="${alert:+$alert; }pipeline finished: $(grep -hE "$done_re" "$RL" | tail -1)"
  if [ -n "$alert" ] && [ "$alert" != "$last_alert" ]; then
    { echo "ALERT $now: $alert"; echo; cat /tmp/hb.txt; echo "--- tail $pfx$stage.log"; tr '\r' '\n' < "$slog" | tail -25; } > /tmp/alert.txt
    cp /tmp/alert.txt "$DRIVE/ALERT.txt"
    # optional phone push: the private ntfy topic lives only in Drive (never in this public repo)
    topic=$(cat "$DRIVE/ntfy_topic.txt" 2>/dev/null)
    [ -n "$topic" ] && curl -s -m 20 -H "Title: Threnody alert" -H "Priority: high" \
      -d "$alert (stage=$stage, gpu=$gpu)" "https://ntfy.sh/$topic" > /dev/null
    last_alert=$alert
  fi
  sleep 300
done
