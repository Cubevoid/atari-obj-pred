#!/bin/sh

# Get a list of all staged Python files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

# Exit if no Python files are staged
if [ -z "$FILES" ]; then
  exit 0
fi

# Run black on each staged Python file
for FILE in $FILES; do
  black --line-length 160 "$FILE"
  git add "$FILE"
done

exit 0