#!/bin/bash

# Poker44 Miner Startup Script

NETUID="${NETUID:-126}"
WALLET_PATH="${WALLET_PATH:-/home/papagruz/.bittensor/wallets}"
WALLET_NAME="${WALLET_NAME:-default}"
HOTKEY="${HOTKEY:-default}"
NETWORK="${NETWORK:-finney}"
SUBTENSOR_CHAIN_ENDPOINT="${SUBTENSOR_CHAIN_ENDPOINT:-ws://192.168.0.116:9951}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PYTHON_BIN="${PYTHON_BIN:-./miner_env/bin/python}"
PM2_NAME="${PM2_NAME:-poker44_miner}"  ##  name of Miner, as you wish
AXON_PORT="${AXON_PORT:-31001}"
AXON_IP="${AXON_IP:-0.0.0.0}"
AXON_EXTERNAL_IP="${AXON_EXTERNAL_IP:-82.199.101.206}"
AXON_EXTERNAL_PORT="${AXON_EXTERNAL_PORT:-$AXON_PORT}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
DRY_RUN="${DRY_RUN:-0}"
POKER44_MODEL_REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-}"

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python interpreter not found or not executable at $PYTHON_BIN"
    echo "Run scripts/miner/setup.sh or set PYTHON_BIN to a valid Python interpreter."
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"

if [ -z "$POKER44_MODEL_REPO_COMMIT" ] && command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null; then
  POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD)"
fi
export POKER44_MODEL_REPO_COMMIT

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.path "$WALLET_PATH"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --subtensor.chain_endpoint "$SUBTENSOR_CHAIN_ENDPOINT"
  --axon.ip "$AXON_IP"
  --axon.port "$AXON_PORT"
  --axon.external_ip "$AXON_EXTERNAL_IP"
  --axon.external_port "$AXON_EXTERNAL_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run only; PM2 process was not started."
  printf 'Command: pm2 start %q --name %q --interpreter %q --' "$MINER_SCRIPT" "$PM2_NAME" "$PYTHON_BIN"
  printf ' %q' "${MINER_ARGS[@]}"
  printf '\n'
  echo "Config: netuid=$NETUID network=$NETWORK chain_endpoint=$SUBTENSOR_CHAIN_ENDPOINT wallet_path=$WALLET_PATH wallet=$WALLET_NAME hotkey=$HOTKEY axon_ip=$AXON_IP axon_port=$AXON_PORT external_ip=$AXON_EXTERNAL_IP external_port=$AXON_EXTERNAL_PORT python=$PYTHON_BIN model_repo_commit=$POKER44_MODEL_REPO_COMMIT"
  exit 0
fi

pm2 start "$MINER_SCRIPT" \
  --name "$PM2_NAME" \
  --interpreter "$PYTHON_BIN" -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK chain_endpoint=$SUBTENSOR_CHAIN_ENDPOINT wallet_path=$WALLET_PATH wallet=$WALLET_NAME hotkey=$HOTKEY axon_ip=$AXON_IP axon_port=$AXON_PORT external_ip=$AXON_EXTERNAL_IP external_port=$AXON_EXTERNAL_PORT python=$PYTHON_BIN model_repo_commit=$POKER44_MODEL_REPO_COMMIT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
