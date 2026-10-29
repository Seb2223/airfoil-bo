#!/usr/bin/env bash
# Install XFoil into tools/ without sudo (Debian/Ubuntu), plus the FPE shim.
set -euo pipefail
cd "$(dirname "$0")"

apt-get download xfoil
dpkg-deb -x xfoil_*.deb xfoil_pkg
cp xfoil_pkg/usr/bin/xfoil ./xfoil
rm -rf xfoil_pkg xfoil_*.deb

# The packaged binary is built with -ffpe-trap, so the viscous solver dies with
# SIGFPE on harmless intermediate NaNs. Preloading nofpe.c turns the trap setup off.
gcc -shared -fPIC -O2 -o libnofpe.so nofpe.c

echo "xfoil installed at $(pwd)/xfoil"
