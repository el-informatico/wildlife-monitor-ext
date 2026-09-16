#!/usr/bin/env bash
# Download the pinned Terraform binary into infra/.tools (gitignored).
# Keeps the repo free of vendored binaries and the version reproducible.
set -euo pipefail

VERSION="1.16.3"
BASEURL="https://releases.hashicorp.com/terraform/${VERSION}"
TOOLS_DIR="$(cd "$(dirname "$0")/../.tools" && mkdir -p . && pwd)"
ZIP="${TOOLS_DIR}/terraform.zip"

if [[ -x "${TOOLS_DIR}/terraform" && "$("${TOOLS_DIR}/terraform" version | head -1)" == "Terraform v${VERSION}" ]]; then
  echo "terraform v${VERSION} already present: ${TOOLS_DIR}/terraform"
  exit 0
fi

echo "downloading terraform ${VERSION}..."
curl -fsSL -o "${ZIP}" "${BASEURL}/terraform_${VERSION}_linux_amd64.zip"

echo "verifying checksum..."
curl -fsSL -o "${TOOLS_DIR}/SHA256SUMS" "${BASEURL}/terraform_${VERSION}_SHA256SUMS"
(cd "${TOOLS_DIR}" && grep ' terraform_.*_linux_amd64.zip$' SHA256SUMS | sha256sum -c -)

python3 -m zipfile -e "${ZIP}" "${TOOLS_DIR}/"
chmod +x "${TOOLS_DIR}/terraform"
"${TOOLS_DIR}/terraform" version
