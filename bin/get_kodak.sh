#!/bin/bash

mkdir -p data/kodak

for i in {01..24..1}; do
  echo ${i}
  wget http://r0k.us/graphics/kodak/kodak/kodim${i}.png -O data/kodak/kodim${i}.png
done