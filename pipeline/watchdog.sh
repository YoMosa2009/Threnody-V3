#!/usr/bin/env bash
# Threnody watchdog (runs beside run_all.sh). Every 5 min it copies a heartbeat to Drive and, on trouble,
# writes ALERT.txt there. Drive FUSE only syncs closed files, so both are written locally then copied.
#   setsid nohup bash pipeline/watchdog.sh > /dev/null 2>&1 &
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
L=$DRIVE/logs; M=$DRIVE/markers
idle=0; last_alert=""
while true; do
  now=$(date -u +%FT%TZ)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)
  util=${gpu##*, }
  procs=$(ps aux | grep -E '[t]rain_sft|[m]erge.py|[b]ench_colab|[l]lama-|[c]onvert_hf|[b]uild_data|[g]en_teacher|[v]llm serve' | awk '{print $12,$13}' | sort -u | tr '\n' ';')
  running=$(pgrep -f 'bash pipeline/[r]un_all' > /dev/null && echo yes || echo no)
  stage=$(grep -o 'start [a-z_0-9]*' "$L/run_all.log" 2>/dev/null | tail -1 | cut -d' ' -f2)
  slog="$L/$stage.log"
  steps=$( { tr '\r' '\n' < "$slog" 2>/dev/null | grep -E "'loss'|'eval_loss'" | tail -2; \
             tr '\r' '\n' < "$slog" 2>/dev/null | grep -E "it/s\]|s/it\]" | tail -1; } )
  # log staleness: minutes since the stage log last changed
  stale=$(( ( $(date +%s) - $(stat -c %Y "$slog" 2>/dev/null || date +%s) ) / 60 ))
  {
    echo "time=$now run_all=$running stage=$stage gpu_mem_MiB,util%=$gpu log_age_min=$stale"
    echo "procs=$procs"
    echo "markers: $(ls $M 2>/dev/null | tr '\n' ' ')"
    echo "--- run_all.log"; tail -5 "$L/run_all.log" 2>/dev/null
    echo "--- $stage.log"; echo "$steps"
  } > /tmp/hb.txt
  cp /tmp/hb.txt "$DRIVE/heartbeat.txt"

  alert=""
  if tr '\r' '\n' < "$slog" 2>/dev/null | tail -40 | grep -qE 'Traceback|Error:|CUDA out of memory|Killed'; then
    alert="ERROR in $stage.log"
  fi
  if [ "$util" -lt 5 ] 2>/dev/null && [ -z "$procs" ]; then idle=$((idle + 1)); else idle=0; fi
  [ $idle -ge 2 ] && ! grep -q 'ALL STAGES FINISHED' "$L/run_all.log" && alert="${alert:+$alert; }GPU idle with no pipeline process"
  [ "$stage" = sft ] && [ $stale -ge 45 ] && alert="${alert:+$alert; }sft log silent ${stale}m"
  grep -q 'ALL STAGES FINISHED' "$L/run_all.log" 2>/dev/null && alert="${alert:+$alert; }run_all finished (eval.done: $( [ -f $M/eval.done ] && echo yes || echo no))"
  if [ -n "$alert" ] && [ "$alert" != "$last_alert" ]; then
    { echo "ALERT $now: $alert"; echo; cat /tmp/hb.txt; echo "--- tail $stage.log"; tr '\r' '\n' < "$slog" | tail -25; } > /tmp/alert.txt
    cp /tmp/alert.txt "$DRIVE/ALERT.txt"
    # optional phone push: the private ntfy topic lives only in Drive (never in this public repo)
    topic=$(cat "$DRIVE/ntfy_topic.txt" 2>/dev/null)
    [ -n "$topic" ] && curl -s -m 20 -H "Title: Threnody alert" -H "Priority: high" \
      -d "$alert (stage=$stage, gpu=$gpu)" "https://ntfy.sh/$topic" > /dev/null
    last_alert=$alert
  fi
  sleep 300
done
