#!/usr/bin/env bash
set -e

API_DIR="$(cd "$(dirname "$0")/external/moltbook-api" && pwd)"

echo "==> Starting Moltbook API dev environment"

# ── 1. Start Postgres ────────────────────────────────────────────────────────
echo "--> Bringing up Postgres (Docker Compose)..."
docker compose -f "$API_DIR/docker-compose.yml" up -d

# ── 2. Wait for Postgres to be healthy ───────────────────────────────────────
echo "--> Waiting for Postgres to be ready..."
until docker compose -f "$API_DIR/docker-compose.yml" exec -T postgres \
    pg_isready -U moltbook -d moltbook -q 2>/dev/null; do
  sleep 1
done
echo "    Postgres is ready."

# ── 3. Start the API server ───────────────────────────────────────────────────
# Schema is applied automatically by Docker on first boot via
# docker-entrypoint-initdb.d/schema.sql — no manual migration needed.
echo "--> Starting Moltbook API (npm run dev)..."
echo "    Logs below — Ctrl-C to stop the API (Postgres keeps running)."
echo ""
cd "$API_DIR" && npm run dev
