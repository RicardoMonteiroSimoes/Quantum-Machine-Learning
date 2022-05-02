#!/usr/bin/env bash
set -e
set -o pipefail

# cd into current script folder
{
  cd "$(dirname "${BASH_SOURCE[0]}")"
} || exit 1

declare -r virtualEnvName="ba-quantum-env"

printf "Local Python paths\n"
printf 'python: %s\n' "$(which python)"
printf 'python3: %s\n' "$(which python3)"
printf 'pip: %s\npip3: %s\n' "$(which pip)" "$(which pip3)"

printf "Install/Upgrade virtualenv\n"
python3 -m pip install virtualenv --upgrade

printf "Activate virtual env \"%s\"\n" "${virtualEnvName}"
# shellcheck disable=SC1090
if [[ -f "${HOME}/envs/${virtualEnvName}/bin/activate" ]]; then
  source "${HOME}/envs/${virtualEnvName}/bin/activate"
else
  python3 -m virtualenv -p "$(which python3)" "${HOME}/envs/${virtualEnvName}"
  source "${HOME}/envs/${virtualEnvName}/bin/activate"
fi

printf "VIRTUAL_ENV: %s\n" "${VIRTUAL_ENV}"
if [[ ! "${VIRTUAL_ENV}" = *${virtualEnvName}* ]]; then
  printf "virtual env \"%s\" is not activated. Aborting\n" "${virtualEnvName}"
  exit 1
fi

printf "Install/Upgrade requirements\n"
if [[ -f "requirements.txt" ]]; then
  pip install -r "requirements.txt" --upgrade
else
  printf "Couldn't find requirements.txt file. Aborting\n"
  exit 1
fi
