#!/usr/bin/env bash
set -e
set -o pipefail

# cd into current script folder
{
  cd "$(dirname "${BASH_SOURCE[0]}")"
} || exit 1

printf "Local Python paths\n"
printf 'python: %s\n' "$(which python)"
printf 'python3: %s\n' "$(which python3)"
printf 'pip: %s\npip3: %s\n' "$(which pip)" "$(which pip3)"

printf "Install/Upgrade requirements\n"
if [[ -f "requirements.txt" ]]; then
  pip install -r "requirements.txt" --upgrade
else
  printf "Couldn't find requirements.txt file. Aborting\n"
  exit 1
fi
