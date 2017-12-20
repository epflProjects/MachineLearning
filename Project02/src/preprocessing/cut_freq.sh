#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat vocab.txt | gsed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f1 > freq.txt
