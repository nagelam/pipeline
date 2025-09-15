#!/bin/bash


SEARCH_PATH="."

echo "delete all .csv и .json in $SEARCH_PATH "
echo "delete? (y/n)"
read -r confirm
if [[ "$confirm" != "y" ]]; then
    echo "exit."
    exit 0
fi

find "$SEARCH_PATH" -maxdepth 2 -type f \( -name "*.csv" -o -name "*.json" \) -print -delete

echo "files deleted."
