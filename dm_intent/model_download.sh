#!/bin/bash

sudo apt-get install unzip

echo "Model Download PATH : "pwd
FILEID='1J9Nui4erfhVlCbcDejnu6LaoNRhCT9Tm'
FILENAME='M2-7_dm_intent.zip'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id="$FILEID > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id="$FILEID -o $FILENAME
rm cookie


rm -rf scripts/authkey/
rm -rf scripts/checkpoints/
rm -rf scripts/data/

unzip $FILENAME

mv DM_intent_hyu/* scripts/

rm $FILENAME

echo "Modelfile Download Complete!!"
