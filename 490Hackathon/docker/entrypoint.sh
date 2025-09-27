
#!/usr/bin/env bash
set -euo pipefail
CMD="${1:-serve}"; shift || true
case "$CMD" in
  train)    exec python -m scripts.train "$@" ;;
  serve)    exec python -m scripts.serve "$@" ;;
  evaluate) exec python -m scripts.evaluate "$@" ;;
  *)        exec "$CMD" "$@" ;;
esac
