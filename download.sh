#!/bin/bash

URL_GAP="https://disk.yandex.ru/d/IuSsWjG6cWhxHA"
DEST_PATH_GAP="models/checkpoint-gap"
URL_SPACE="https://disk.yandex.ru/d/-eP1roOslU5hsQ"
DEST_PATH_SPACE="models/checkpoint-space"

function download_checkpoint {
    URL="$1"
    DEST_PATH="$2"
    NAME="$(basename "$DEST_PATH")"
    ZIP="$NAME".zip
    DIR="$(dirname "$DEST_PATH")"

    if [ ! -d "$DEST_PATH" ]; then
        echo "Downloading $NAME..."
        mkdir -p "$(dirname "$DEST_PATH")"
        if ! ./yd-wget.py "$URL"; then
            echo "Error: fail do download $NAME"
            return 1
        fi
        unzip "$ZIP" -d "$DIR"
        rm "$ZIP"
    else
        echo "$NAME already downloaded"
    fi
}

download_checkpoint "$URL_GAP" "$DEST_PATH_GAP"
download_checkpoint "$URL_SPACE" "$DEST_PATH_SPACE"
