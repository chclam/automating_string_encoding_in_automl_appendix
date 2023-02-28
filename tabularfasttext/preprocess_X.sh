#!/usr/bin/env sh
sed "s/[^a-zA-Z0-9_.]/ /g; s/None/ /g; s/[ ][ ]*/ /g; s/\\n/ /g;" $1 | tr "[A-Z]" "[a-z]" > $1.tmp && rm $1 && mv $1.tmp $1

