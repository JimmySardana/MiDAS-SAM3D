#!/bin/bash
#SBATCH --job-name=sam3d-cluster
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=40
#SBATCH --mem=128G
#SBATCH --time=00:60:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --signal=B:TERM@30

# ---------------- USER CONFIG ----------------
IMAGE_NAME="sam3d"
IMAGE="pc3163.igd.fraunhofer.de:4567/${IMAGE_NAME}"

LOGIN_HOST="amperecontrol1"
LOGIN_USER="sardana"
SSH_KEY="${HOME}/.ssh/id_ed25519_cluster"   # must work without password

CONTAINER_PORT=8000                          # port inside container
HOST_BIND_IP="127.0.0.1"                     # bind on compute node (local only)
HOST_PORT=8000                               # port on compute node


LOGIN_PORT=18000

CONTAINER_NAME="${IMAGE_NAME}-${SLURM_JOB_ID}"
DATAPATH="/opt/cache/sardana/"
CONFIG_PATH="${DATAPATH}/sam3d_checkpoint"

# ---------------- HELPERS ----------------
log() { echo "[$(date +'%F %T')] $*"; }

cleanup() {
  log "CLEANUP: starting..."

  # Kill the reverse tunnel first so login port frees immediately
  if [[ -n "${TUNNEL_PID:-}" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    log "CLEANUP: stopping SSH tunnel (pid=$TUNNEL_PID)"
    kill "$TUNNEL_PID" 2>/dev/null || true
    wait "$TUNNEL_PID" 2>/dev/null || true
  fi

  # Stop docker container (best) then kill docker run wrapper if needed
  log "CLEANUP: stopping container ${CONTAINER_NAME} (if running)"
  docker stop "$CONTAINER_NAME" 2>/dev/null || true

  if [[ -n "${DOCKER_PID:-}" ]] && kill -0 "$DOCKER_PID" 2>/dev/null; then
    log "CLEANUP: stopping docker run process (pid=$DOCKER_PID)"
    kill "$DOCKER_PID" 2>/dev/null || true
    wait "$DOCKER_PID" 2>/dev/null || true
  fi

  log "CLEANUP: done."
}

gio_mount_nas()
{
    mkdir -p ~/dbus
    XDG_RUNTIME_DIR=~/dbus
    export DBUS_SESSION_BUS_ADDRESS=`dbus-daemon --fork --print-address --session`
    gio mount smb://pc3163/nobackup < ~/smbcreds
}

on_term() {
  log "SIGTERM received (likely scancel or time limit). Cleaning up..."
  cleanup
  exit 0
}

trap on_term TERM
trap cleanup EXIT INT

# ---------------- PRECHECKS ----------------
log "Job started on node: $(hostname)"
log "Image: $IMAGE"
log "Compute bind: ${HOST_BIND_IP}:${HOST_PORT} -> container:${CONTAINER_PORT}"
log "Reverse tunnel target: ${LOGIN_HOST}:${LOGIN_PORT} -> ${HOST_BIND_IP}:${HOST_PORT}"



# ---------------- COPY FROM NAS ----------------
if test ! -d $DATAPATH/sam3d_checkpoint ; then
    # mount NAS
    gio_mount_nas
 
    mkdir -p $DATAPATH
    cd $DATAPATH
    # copy necessary data
    gio copy --progress smb://pc3163/nobackup/spool/sam3d_checkpoint.zip $DATAPATH
    unzip -q sam3d_checkpoint.zip -d sam3d_checkpoint
fi
 
# ---------------- RUN DOCKER IMAGE ----------------
log "Starting container in background..."
rootless-docker run --gpus all \
    --name "$CONTAINER_NAME" \
    -p "${HOST_BIND_IP}:${HOST_PORT}:${CONTAINER_PORT}" \
    -e SAM3D_CONFIG="/checkpoints/pipeline.yaml" \
    -v "${CONFIG_PATH}:/checkpoints" \
    "$IMAGE" &
DOCKER_PID=$!
log "Container started (docker pid=$DOCKER_PID)"

# ---------------- START REVERSE TUNNEL ----------------
log "Starting reverse tunnel (compute -> login) ..."
ssh -N \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -i "$SSH_KEY" \
  -R "${LOGIN_PORT}:${HOST_BIND_IP}:${HOST_PORT}" \
  "${LOGIN_USER}@${LOGIN_HOST}" &
TUNNEL_PID=$!
log "Reverse tunnel started (pid=$TUNNEL_PID)"

# ---------------- WAIT ----------------
# If either dies, end the job (and cleanup will run).
while true; do
  if ! kill -0 "$DOCKER_PID" 2>/dev/null; then
    log "Docker process exited. Ending job."
    exit 0
  fi
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    log "Tunnel process exited. Ending job."
    exit 0
  fi
  sleep 2
done




