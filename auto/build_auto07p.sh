#!/usr/bin/env bash
# Build auto-07p locally (no sudo) into ~/auto-07p. Prereqs: gfortran, make, git.
# Run inside WSL:  bash auto/build_auto07p.sh
set -euo pipefail

AUTO_HOME="${HOME}/auto-07p"

if ! command -v gfortran >/dev/null || ! command -v make >/dev/null; then
  echo "ERROR: gfortran/make missing. Run:  sudo apt install -y gfortran make" >&2
  exit 1
fi

if [ ! -d "${AUTO_HOME}/.git" ]; then
  echo "== cloning auto-07p =="
  git clone --depth 1 https://github.com/auto-07p/auto-07p.git "${AUTO_HOME}"
fi

cd "${AUTO_HOME}"
echo "== configure =="
./configure --enable-plaut=no --enable-plaut04=no >/tmp/auto_configure.log 2>&1 || ./configure
echo "== make =="
make >/tmp/auto_make.log 2>&1
echo "== done; AUTO built at ${AUTO_HOME} =="
echo "Add to your shell (and CI/driver):  source ${AUTO_HOME}/cmds/auto.env.sh"
ls -1 "${AUTO_HOME}/cmds/auto.env.sh" "${AUTO_HOME}/bin/"* 2>/dev/null | head
