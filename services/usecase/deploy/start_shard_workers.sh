#!/usr/bin/env bash
#
# start_shard_workers.sh — launch one Celery worker per camera shard.
#
# Each worker runs --concurrency=1 and consumes exactly one queue
# (usecase_shard_<i>), so a camera pinned to that shard has its frames
# processed strictly in order while other cameras run in parallel on other
# shards. See workers/queue.py (_shard_for / N_SHARDS) and
# workers/tasks.evaluate_frame_task.
#
# N_SHARDS MUST match the value the API process uses (workers/queue.py reads
# the same env var) — otherwise the producer may route to a shard with no
# worker.
#
# Usage:
#   N_SHARDS=5 ./deploy/start_shard_workers.sh          # foreground (celery multi, backgrounds workers)
#   N_SHARDS=5 ./deploy/start_shard_workers.sh stop     # stop the workers
#
# Run this from the service root: services/usecase/
set -euo pipefail

N_SHARDS="${N_SHARDS:-5}"
LOGLEVEL="${CELERY_LOGLEVEL:-info}"
PIDDIR="${CELERY_PIDDIR:-/tmp/sakshi-usecase-workers}"
LOGDIR="${CELERY_LOGDIR:-logs}"
CELERY_BIN="${CELERY_BIN:-celery}"

mkdir -p "$PIDDIR" "$LOGDIR"

# Build the `celery multi` node list: one node per shard, each pinned to its
# own queue via -Q:<node>.
nodes=()
queue_args=()
for ((i = 0; i < N_SHARDS; i++)); do
  nodes+=("shard${i}")
  queue_args+=("-Q:shard${i}" "usecase_shard_${i}")
done

action="${1:-start}"

case "$action" in
  start|restart)
    echo "[start_shard_workers] $action $N_SHARDS shard workers (concurrency=1 each)"
    "$CELERY_BIN" -A workers.celery_app multi "$action" "${nodes[@]}" \
      --concurrency=1 \
      --loglevel="$LOGLEVEL" \
      --pidfile="${PIDDIR}/%n.pid" \
      --logfile="${LOGDIR}/celery-%n.log" \
      "${queue_args[@]}"
    ;;
  stop|stopwait)
    echo "[start_shard_workers] stopping $N_SHARDS shard workers"
    "$CELERY_BIN" -A workers.celery_app multi "$action" "${nodes[@]}" \
      --pidfile="${PIDDIR}/%n.pid"
    ;;
  *)
    echo "usage: $0 [start|restart|stop|stopwait]" >&2
    exit 2
    ;;
esac
