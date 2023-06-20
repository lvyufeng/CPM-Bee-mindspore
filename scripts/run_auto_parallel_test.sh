
#!/bin/bash

# Get the first argument (integer)
number=$1
script=$2
# Validate the input
re='^[0-9]+$'
if ! [[ $number =~ $re ]]; then
  echo "Invalid argument. Please provide a valid integer."
  exit 1
fi

mpirun --allow-run-as-root -n $number pytest -v -s $script