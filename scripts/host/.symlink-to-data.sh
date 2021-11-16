#!/bin/bash

DATA_DIR="../../data"

if [ ! -d $DATA_DIR ]; then
    echo "Creating an data directory.\n"
    mkdir -p $DATA_DIR
else
    echo "Data directory [$DATA_DIR] already exists.\n"
fi

LINK_FILE="data"

if [ -L $LINK_FILE ]; then
    echo "Symbolic link [$LINK_FILE] exists already.\n"
else
    echo "Making symbolic link [$LINK_FILE] to [$DATA_DIR].\n"
    ln -s $DATA_DIR $LINK_FILE
fi
