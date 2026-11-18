#!/usr/bin/env bash
set -e

for script in "$(dirname "$0")"/eval_class_*.sh; do
    bash "$script" "$@"
done
