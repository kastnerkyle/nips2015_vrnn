#!/bin/bash

DIRNAME=blizzard_wav
if [ ! -d "$DIRNAME" ]; then
  mkdir $DIRNAME
  fi

echo $files_list
for f in `cat list_of_files.txt | grep .mp3 | sed -e 's/[ \t][ \t]*/ /g' | cut -d ' ' -f6`; do
    t=`basename $f`
    ffmpeg -i "$f" -acodec pcm_s16le -ac 1 -ar 16000 $DIRNAME/${t%.mp3}.wav
done
