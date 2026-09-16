#!/usr/bin/env bash
# Build the custom Lambda execution image (python3.12 + source-built OpenCV 5).
#
#   WFM_PIPELINE_VENV=/path/to/sprints-1-2/venv ./scripts/build-lambda-image.sh
#
# The venv must be python 3.12 and contain the source-built cv2 5.0.0 (the
# OpenCV the project builds and runs locally). The script harvests:
#
#   - the cv2 python package (binding wrapper + loader), with the two
#     cmake-generated config files rewritten to portable relative paths
#     (upstream bakes the build venv's absolute location into them);
#   - the libopencv_*.so.5.0.0 shared libraries the wrapper links against.
#
# Context and image land under infra/build/ (gitignored). The image must
# then be wired into LocalStack via LAMBDA_RUNTIME_IMAGE_MAPPING — see
# .env.example and README.md ("MOG2 inside the Lambda"). The compose stack's
# 02- localstack-init hook makes LocalStack accept a runtime image that
# exists only in the local docker daemon.
set -euo pipefail

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${WFM_LAMBDA_IMAGE_TAG:-wfm-lambda-py312-opencv5:g1}"
VENV="${WFM_PIPELINE_VENV:-}"

if [ -z "$VENV" ]; then
  echo "error: WFM_PIPELINE_VENV is not set — point it at the sprints-1-2" >&2
  echo "       pipeline venv (python 3.12, source-built OpenCV 5)." >&2
  exit 2
fi
VENV="$(cd "$VENV" && pwd)"

SITE="$VENV/lib/python3.12/site-packages"
LIBDIR="$VENV/lib"
for req in "$SITE/cv2/__init__.py" "$LIBDIR/libopencv_core.so.5.0.0"; do
  if [ ! -f "$req" ]; then
    echo "error: $req not found — is this the pipeline venv with the OpenCV 5 build?" >&2
    exit 2
  fi
done

if ! "$VENV/bin/python" -c 'import sys, cv2; assert sys.version_info[:2] == (3, 12), sys.version; assert cv2.__version__.startswith("5."), cv2.__version__' 2>/dev/null; then
  echo "error: the venv's cv2 is not importable OpenCV 5 on python 3.12" >&2
  exit 2
fi

CONTEXT="$INFRA_DIR/build/lambda-image"
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT/opencv/lib"

# --- cv2 package, config files rewritten to portable paths -----------------
cp -a "$SITE/cv2" "$CONTEXT/opencv/cv2"
find "$CONTEXT/opencv/cv2" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# config.py: where the shared libopencv_*.so live. Layout in the image:
# /opt/python/cv2 + /opt/opencv-5.0.0/lib -> ../../opencv-5.0.0/lib from the
# package dir. (BINARIES_PATHS/PYTHON_EXTENSIONS_PATHS are pre-initialized
# by the cv2 loader before it execs these files, so appending works.)
cat > "$CONTEXT/opencv/cv2/config.py" <<'EOF'
import os
BINARIES_PATHS = [
    os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'opencv-5.0.0', 'lib')),
] + BINARIES_PATHS
EOF

cat > "$CONTEXT/opencv/cv2/config-3.12.py" <<'EOF'
import os
PYTHON_EXTENSIONS_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python-3.12'),
] + PYTHON_EXTENSIONS_PATHS
EOF

# --- shared libraries (the wrapper's DT_NEEDED set, dnn included) ----------
for lib in core imgproc imgcodecs video flann geometry dnn; do
  cp -L "$LIBDIR/libopencv_${lib}.so.500" "$CONTEXT/opencv/lib/libopencv_${lib}.so.5.0.0"
  ln -s "libopencv_${lib}.so.5.0.0" "$CONTEXT/opencv/lib/libopencv_${lib}.so.500"
done

# --- runtime bootstrap files (committed alongside the Dockerfile) ----------
cp -a "$INFRA_DIR/docker/runtime" "$CONTEXT/runtime"

echo "--- context prepared at $CONTEXT"
du -sh "$CONTEXT"

docker build -f "$INFRA_DIR/docker/Dockerfile.lambda-py312-opencv5" -t "$IMAGE_TAG" "$CONTEXT"

echo "--- image built: $IMAGE_TAG"
echo "Smoke-test the bindings in-image (PYTHONPATH is set by the runtime bootstrap):"
docker run --rm --entrypoint /var/lang/bin/python3.12 -e PYTHONPATH=/opt/python "$IMAGE_TAG" -c \
  'import cv2; print("cv2", cv2.__version__, "@", cv2.__file__); m = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=24.0); print("MOG2 OK")'

echo "Set in .env for docker compose (then restart the stack):"
echo "  LAMBDA_RUNTIME_IMAGE_MAPPING={\"python3.12\": \"$IMAGE_TAG\"}"
