#!/bin/bash
# LocalStack pulls a function's runtime image from a registry at
# function-create time (docker_runtime_executor._ensure_runtime_image_present)
# and treats a failed pull as fatal: the function ends up in state Failed
# ("Error while creating lambda: <image>"). There is no "image already exists
# in the local daemon" fast path, so runtime images that were never pushed to
# any registry — like this repo's custom python3.12 + OpenCV 5 image built by
# scripts/build-lambda-image.sh and referenced through
# LAMBDA_RUNTIME_IMAGE_MAPPING — can never create.
#
# This hook patches that one function (inside this ephemeral container only)
# to skip the pull when the image already exists in the docker daemon
# LocalStack is connected to. Registry images behave exactly as before.
set -euo pipefail

FILE=/opt/code/localstack/localstack-core/localstack/services/lambda_/invocation/docker_runtime_executor.py

python3 - "$FILE" <<'EOF'
import sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = """        try:
            CONTAINER_CLIENT.pull_image(image, platform)
            PULLED_IMAGES.add((image, platform))
"""
new = """        try:
            # patch (this repo): accept images already present in the local
            # docker daemon instead of failing the create on a registry pull
            if image not in CONTAINER_CLIENT.get_docker_image_names(
                include_tags=True, strip_wellknown_repo_prefixes=False
            ):
                CONTAINER_CLIENT.pull_image(image, platform)
            PULLED_IMAGES.add((image, platform))
"""

count = src.count(old)
if count != 1:
    sys.exit(
        f"expected exactly one patch target in {path}, found {count} — "
        "LocalStack internals changed; update this hook"
    )

with open(path, "w") as f:
    f.write(src.replace(old, new))
EOF

echo "lambda runtime images: local daemon images accepted (patched _ensure_runtime_image_present)"
