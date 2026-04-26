#!/bin/bash
set -euo pipefail

exec "$(dirname "$0")/train/train_fd.sh" "$@"
