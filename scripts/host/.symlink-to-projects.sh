#!/bin/bash

DATA_DIR="../../projects"

if [ ! -d $DATA_DIR ]; then
    echo "Creating an projects directory.\n"
    mkdir -p $DATA_DIR
else
    echo "Project directory [$DATA_DIR] already exists.\n"
fi

LINK_FILE="projects"

if [ -L $LINK_FILE ]; then
    echo "Symbolic link [$LINK_FILE] exists already.\n"
else
    echo "Making symbolic link [$LINK_FILE] to [$DATA_DIR].\n"
    ln -s $DATA_DIR $LINK_FILE
fi
