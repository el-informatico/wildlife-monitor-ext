#!/bin/sh
# Runs inside the LocalStack container at startup (ready.d hook).
#
# Docker daemons configured with `userns-remap` (as this host's is) reject
# `put_archive` of files owned by uids outside the subordinate range: the
# lambda runtime under /usr/lib/localstack/lambda-runtime ships owned by uid
# 101000, and copying it into a remapped execution-environment container
# fails with "container ID 101000 cannot be mapped to a host ID". LocalStack
# surfaces that as a startup failure with an empty error message and retries
# forever — no lambda ever runs.
#
# This container runs with userns_mode: host (see docker-compose.yml), so
# root here is real root: re-owning the runtime to uid/gid 0 (always
# mappable) fixes the copy. Harmless on non-remapping daemons.
set -eu
chown -R 0:0 /usr/lib/localstack/lambda-runtime
echo "lambda runtime re-owned to 0:0 (userns-remap compatibility)"
