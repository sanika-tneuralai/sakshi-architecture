#!/usr/bin/env bash
set -euo pipefail

UNIT=sakshi-alert.service
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$UNIT"

sudo cp "$SRC" /etc/systemd/system/$UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now "$UNIT"
sudo systemctl status --no-pager "$UNIT" || true

echo
echo "Installed $UNIT."
echo "  logs: journalctl -u $UNIT -f"
echo "  restart: sudo systemctl restart $UNIT"
