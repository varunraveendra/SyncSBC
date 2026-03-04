#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${1:?Usage: $0 path/to/script.py}"

# Known listeners: 8226 (Code Editor executor), 8193 (Debug Python Shell)
PORTS=(8226 8193)

for P in "${PORTS[@]}"; do
  if nc -z 127.0.0.1 "$P" 2>/dev/null; then
    echo "➡️  Sending $SCRIPT to Isaac Sim on port $P via raw TCP..."
    # Send file contents to the socket
    nc 127.0.0.1 "$P" < "$SCRIPT"
    echo "✅ Sent."
    exit 0
  fi
done

cat <<EOF
❌ No known Isaac Sim TCP listeners found (tried 8226, 8193).

Fix:
1) If you use the built-in Code Editor, make sure its executor is running.
2) Or enable the Debug Python Shell:
   - Window → Extensions → enable "omni.kit.debug.python_shell"
   - (Default port is usually 8193)
Then rerun this script.
EOF
exit 1
