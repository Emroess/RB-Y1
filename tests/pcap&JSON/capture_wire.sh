#!/usr/bin/env bash
#
# Capture the raw gRPC/HTTP2 wire traffic between this U-PC and the RB-Y1 R-PC.
# This records the ACTUAL bytes the robot emits (TCP -> HTTP/2 -> gRPC-framed
# protobuf), unlike the SDK which hands you already-decoded objects.
#
# Usage:
#   ./capture_wire.sh [ROBOT_IP] [PORT] [OUTFILE]
#   sudo is required (raw packet capture). Run your SDK client while this runs.
#
# Example:
#   ./capture_wire.sh 192.168.30.1 50051 rby1_wire.pcap
#   # ... in another terminal, run read_state_sdk.py ...
#   # Ctrl-C here to stop.

set -euo pipefail

ROBOT_IP="${1:-192.168.30.1}"
PORT="${2:-50051}"
OUT="${3:-rby1_wire_$(date +%Y%m%d_%H%M%S).pcap}"

# Find the interface with a route to the robot.
IFACE="$(ip route get "$ROBOT_IP" 2>/dev/null \
        | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
if [[ -z "${IFACE:-}" ]]; then
  echo "Could not auto-detect interface to $ROBOT_IP. List with: ip -br addr" >&2
  exit 1
fi

print_decode_hints() {
  cat <<EOF

Capture written to: $OUT
----------------------------------------------------------------------
To read it (the bytes are framed: TCP -> HTTP/2 frames -> gRPC messages
[1 flag byte + 4-byte big-endian length + protobuf] -> protobuf fields):

  Easiest (GUI):
    wireshark $OUT
    # Wireshark auto-dissects HTTP/2 + gRPC. For NAMED protobuf fields,
    # point it at Rainbow's .proto: Preferences > Protobuf > search paths,
    # and enable the gRPC dissector. Without the .proto you still see the
    # protobuf field numbers + wire types.

  Headless (tshark):
    tshark -r $OUT -d tcp.port==$PORT,http2 -Y grpc -O http2,grpc,protobuf

  Schema-free single message (if you extract one gRPC payload to msg.bin):
    protoc --decode_raw < msg.bin    # dumps field#/wire-type tree, no .proto
----------------------------------------------------------------------
EOF
}
trap print_decode_hints EXIT

echo "Interface : $IFACE"
echo "Filter    : tcp port $PORT and host $ROBOT_IP"
echo "Output    : $OUT"
echo "Capturing... run your SDK client now, Ctrl-C to stop."
echo

# -s 0 : full packets (don't truncate)  | host+port filter trims noise
sudo tcpdump -i "$IFACE" -s 0 -w "$OUT" "tcp port $PORT and host $ROBOT_IP"
