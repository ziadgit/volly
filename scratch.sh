set -a; source .env; set +a; \
  rm -f /tmp/tier-*.json; \
  for i in 1 2 3 4 5 6; do \
    curl -s -o "/tmp/tier-$i.json" -X POST \
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=$GEMINI_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"contents":[{"parts":[{"text":"hi"}]}],"generationConfig":{"maxOutputTokens":1}}' & \
  done; wait; \
  echo "=== 200s: $(grep -l '\"text\"' /tmp/tier-*.json 2>/dev/null | wc -l)/6 ==="; \
  echo "=== quotaId labels (should be EMPTY for Tier 1): ==="; \
  grep -ho '"quotaId"[^,]*' /tmp/tier-*.json 2>/dev/null | sort -u
