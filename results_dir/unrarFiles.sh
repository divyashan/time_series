#!/bin/bash

# this uses unrar, which I believe is a unix-only tool

inDir=$1
outDir=$2

function usage() {
	echo 'usage: unrarFiles srcDir destDir'
}

# ================================================================
# making sure we don't do terrible things

if [ "$#" -ne 2 ]; then
	echo "Error: requires 2 arguments"
    usage
    exit
fi

if [ ! -d "$inDir" ]; then
	echo "first arg must be existing directory"
    usage
fi

if [ ! -d "$outDir" ]; then
	mkdir -p "$outDir"
	echo "creating output directory $outDir"
    usage
fi

# ================================================================
# actual functionality

# separate by line, not whitespace, in case there's whitespace in the filenames
IFS='
'

for f in $(ls "${inDir}"/*.rar); do
	# echo $f
	# unrar t "$f"
	fname=$(basename "$f" .rar)
	subdir="$outDir/$fname/"
	if [ ! -d "$subdir" ]; then
		mkdir -p "$subdir"
	fi

	echo "extracting to $subdir..."
	unrar e "$f" "$subdir"
done

unset IFS