#!/usr/bin/env bash
set -e
status=0
for d in */ ; do
    d=${d%/}
    if [[ "$d" == .* ]]; then
        continue
    fi
    if ls "$d"/*.py >/dev/null 2>&1; then
        if [ ! -f "$d/__init__.py" ]; then
            echo "Missing __init__.py in $d" >&2
            status=1
        fi
    fi
    # also check immediate subdirectories within $d? But instruction says top-level only
    # So we won't check deeper
done
exit $status
