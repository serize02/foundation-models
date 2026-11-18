#!/usr/bin/env bash
set -e

for script in "$(dirname "$0")"/etl_*.sh; do
    bash "$script" "$@"
done
