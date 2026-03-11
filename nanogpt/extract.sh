#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./extract_train_gpt2_versions.sh /path/to/repo [start_commit]
#
# Outputs (in current directory):
#   train_gpt2_0001_<shortsha>.py
#   train_gpt2_0002_<shortsha>.py
#   ...
#   train_gpt2_manifest.tsv

REPO="${1:-.}"
START="${2:-9ac321e299f06c0414bd577fd2776f4562283a8a}"

# Set FIRST_PARENT=0 in your environment if you want *all* reachable commits,
# not just the mainline history.
FIRST_PARENT="${FIRST_PARENT:-1}"

OUT_PREFIX="${OUT_PREFIX:-train_gpt2}"
MANIFEST="${OUT_PREFIX}_manifest.tsv"
PAD="${PAD:-4}"

git_in_repo() { git -C "$REPO" "$@"; }

# --- sanity checks ---
git_in_repo rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "Error: '$REPO' is not a git working tree." >&2
  exit 1
}

git_in_repo cat-file -e "${START}^{commit}" >/dev/null 2>&1 || {
  echo "Error: start commit '$START' not found in repo '$REPO'." >&2
  exit 1
}

git_in_repo merge-base --is-ancestor "$START" HEAD >/dev/null 2>&1 || {
  echo "Error: start commit '$START' is not an ancestor of HEAD in '$REPO'." >&2
  echo "       Checkout the branch you want (e.g., main) and re-run." >&2
  exit 1
}

# Include START itself. If START has a parent, use START^..HEAD; else fall back.
if git_in_repo rev-parse "${START}^" >/dev/null 2>&1; then
  RANGE="${START}^..HEAD"
else
  RANGE="${START}..HEAD"
fi

# --- build commit list ---
REVLIST_ARGS=(rev-list --reverse)
if [[ "$FIRST_PARENT" == "1" ]]; then
  REVLIST_ARGS+=(--first-parent)
fi
REVLIST_ARGS+=("$RANGE")

# --- write manifest header ---
printf "index\toutfile\tcommit_sha\tshort_sha\tpath\tcommit_date\tsubject\n" > "$MANIFEST"

i=0
while IFS= read -r sha; do
  [[ -z "$sha" ]] && continue
  short="${sha:0:12}"

  # Find any path(s) named train_gpt2.py in this commit
  mapfile -t paths < <(git_in_repo ls-tree -r --name-only "$sha" | grep -E '(^|/)train_gpt2\.py$' || true)

  # If none in this commit, skip
  (( ${#paths[@]} == 0 )) && continue

  date="$(git_in_repo show -s --format=%cI "$sha")"
  subj="$(git_in_repo show -s --format=%s "$sha")"
  # sanitize tabs just in case
  subj="${subj//$'\t'/ }"

  for path in "${paths[@]}"; do
    ((i++))
    printf -v idx "%0${PAD}d" "$i"
    outfile="${OUT_PREFIX}_${idx}_${short}.py"

    # Extract file content from that commit without checking out
    git_in_repo show "${sha}:${path}" > "$outfile"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$idx" "$outfile" "$sha" "$short" "$path" "$date" "$subj" >> "$MANIFEST"
  done
done < <(git_in_repo "${REVLIST_ARGS[@]}")

echo "Done. Wrote $i file(s) to: $(pwd)"
echo "Manifest: $(pwd)/$MANIFEST"
