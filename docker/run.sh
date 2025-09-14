#!/bin/bash
# Simplified launcher: SMPL-X only (no nosmplx / no sam2)
set -e

export UNAME=${UNAME:-$(whoami)}
export USERID=${USERID:-$(id -u)}
export GID=${GID:-$(id -g)}
export PASSWORD=${PASSWORD:-password}

SERVICE_NAME="dev_smplx"
CONTAINER_NAME="drivestudio-${UNAME}-dev-smplx"

echo "Starting DriveStudio (SMPL-X only)"
echo "User: $UNAME (uid=$USERID gid=$GID)"

cd "$(dirname "$0")"
docker compose -p "$UNAME" up -d "$SERVICE_NAME"

echo "Container started!"
echo "docker exec -it $CONTAINER_NAME bash"
