#!/usr/bin/env bash

# pre-commit.sh

git stash -q --keep-index

# Test prospective commit
python -m unittest discover tests/fast
RESULT=$?

git stash pop -q

[ $RESULT -ne 0 ] && exit 1
exit 0
