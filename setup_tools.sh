#!/usr/bin/env bash
# Download plink2 binary into ./tools/
# Cross-platform: Linux x86_64, macOS arm64/x86_64, Windows (use Git Bash / WSL).
set -euo pipefail

mkdir -p tools
cd tools

UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

# Latest stable plink2 build (as of 2026-05). Update version as needed.
PLINK_VERSION="20250129"

case "${UNAME_S}_${UNAME_M}" in
  Linux_x86_64)
    URL="https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_${PLINK_VERSION}.zip"
    ;;
  Linux_aarch64)
    URL="https://s3.amazonaws.com/plink2-assets/plink2_linux_arm64_${PLINK_VERSION}.zip"
    ;;
  Darwin_arm64)
    URL="https://s3.amazonaws.com/plink2-assets/plink2_mac_arm64_${PLINK_VERSION}.zip"
    ;;
  Darwin_x86_64)
    URL="https://s3.amazonaws.com/plink2-assets/plink2_mac_${PLINK_VERSION}.zip"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    URL="https://s3.amazonaws.com/plink2-assets/plink2_win64_${PLINK_VERSION}.zip"
    ;;
  *)
    echo "Unknown platform ${UNAME_S}_${UNAME_M} — see https://www.cog-genomics.org/plink/2.0/"
    exit 1
    ;;
esac

echo "Downloading plink2 from $URL ..."
curl -fLO "$URL"
ZIP=$(basename "$URL")
unzip -o "$ZIP"
rm "$ZIP"
chmod +x plink2 plink2.exe 2>/dev/null || true
echo "Installed plink2:"
./plink2 --version 2>/dev/null || ./plink2.exe --version
