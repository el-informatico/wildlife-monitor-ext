#!/usr/bin/env bash
# Sensitive-content guard: refuses a commit whose STAGED ADDED LINES
# contain configured sensitive tokens — host paths, Windows-mount paths,
# sibling-project identifiers. Invoked by scripts/hooks/pre-commit.
#
# TOKEN LIST: literal fixed strings, one per line, blank lines ignored,
# read from $(git rev-parse --absolute-git-dir)/sensitive-tokens —
# inside the git dir, never tracked. Re-seed the list with a plain copy
# after a fresh clone.
#
# EXIT CODES: violation => 1; guard-internal failure (no git dir, mktemp
# failure, tokens file present but unreadable) => 2 (fail closed);
# check disabled (tokens file absent/empty) => 0.
set -u

GIT_DIR="$(git rev-parse --absolute-git-dir)" || {
  echo "guard-sensitive-content: cannot determine git dir" >&2
  exit 2
}
TOKENS="$GIT_DIR/sensitive-tokens"

if [ ! -e "$TOKENS" ] && [ ! -L "$TOKENS" ]; then
  echo "guard-sensitive-content: no sensitive-tokens list in the git dir; check disabled" >&2
  exit 0
fi
if [ ! -f "$TOKENS" ] || [ ! -r "$TOKENS" ]; then
  echo "guard-sensitive-content: sensitive-tokens exists but is not a readable file; failing closed" >&2
  exit 2
fi

PATTERNS="$(mktemp)" || {
  echo "guard-sensitive-content: mktemp failed; failing closed" >&2
  exit 2
}
trap 'rm -f "$PATTERNS"' EXIT
# Strip CR and blanks — a blank pattern would match everything.
if ! tr -d '\r' < "$TOKENS" | sed -e '/^[[:space:]]*$/d' > "$PATTERNS"; then
  echo "guard-sensitive-content: cannot sanitize token list; failing closed" >&2
  exit 2
fi
if [ ! -s "$PATTERNS" ]; then
  echo "guard-sensitive-content: sensitive-tokens carries no usable lines; check disabled" >&2
  exit 0
fi

fail=0
blocked=0
while IFS= read -r -d '' f; do
  # Added lines only, taken from INSIDE the hunks; strip the leading '+'.
  added="$(git diff --cached -U0 -- "$f" \
           | awk '/^@@/{inhunk=1; next} inhunk && /^\+/{sub(/^./,""); print}')"
  [ -n "$added" ] || continue
  if printf '%s\n' "$added" | grep -aFqf "$PATTERNS"; then
    echo "guard-sensitive-content: BLOCKED — staged additions in $f" >&2
    echo "  contain configured sensitive tokens." >&2
    fail=1
    blocked=$((blocked + 1))
  fi
done < <(git diff --cached --name-only -z --diff-filter=ACMR)

if [ "$fail" -ne 0 ]; then
  echo "  Rewrite the flagged additions with tokenized references and" >&2
  echo "  re-stage; do not use --no-verify." >&2
  exit 1
fi
exit 0
