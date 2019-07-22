#!/usr/bin/env bash

# Download scripts
curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py

rm -rf train test

for dir in "train" "test"; do
    for line in $(cat ../../Data/Synthetic/Dots/${dir}/${dir}.csv); do
        id=$(echo $line | awk -F, '{print $2}')
        if [ $id != "\"imageId\"" ]; then
            numDots=$(echo $line | awk -F, '{print $3}')
            mkdir -p ${dir}/$numDots
            echo "copying ../../Data/Synthetic/Dots/${dir}/img${id}.jpg to ${dir}/$numDots"
            cp ../../Data/Synthetic/Dots/${dir}/img${id}.jpg ${dir}/$numDots
        fi
    done
done

echo "retrain model..."
rm -rf output; mkdir -p output/intermediate; mkdir -p output/summaries
python retrain.py --image_dir=train \
       --summary_dir=output/summaries --saved_model_dir=output/model \
       --output_labels=output/labels.txt --output_graph=output/graph.pb

# Check how well the training performed
echo "to check performance, type tensorboard --logdir=/tmp/retrain_logs/"

echo "test the training..."
varError="0"
numFailures="0"
numPredictions="0"
for line in $(cat ../../Data/Synthetic/Dots/test/test.csv); do
  id=$(echo $line | awk -F, '{print $2}')
  if [ $id != "\"imageId\"" ]; then
    numDots=$(echo $line | awk -F, '{print $3}')
    # classify
    echo "classifying img${id}.jpg"
    python label_image.py --image=../../Data/Synthetic/Dots/test/img${id}.jpg \
       --graph=output/graph.pb --labels=output/labels.txt \
       --input_layer=Placeholder --output_layer=final_result > result.txt
    # find the most likely label
    gotNumDots=$(python findNumDots.py result.txt)
    diffSquare=$(python -c "print(($numDots - $gotNumDots)**2)")
    if [ $diffSquare != "0" ]; then
      echo "found $gotNumDots in img${id}.jpg but there were $numDots dots"
      numFailures=$(python -c "print($numFailures + 1)")
    else
      echo "found $numDots dots (correct)"
    fi
    # update the score
    varError=$(python -c "print($varError + $diffSquare)")
  fi
  numPredictions=$(python -c "print($numPredictions + 1)")
done
percentFailures=$(python -c "print(100*$numFailures/$numPredictions")
echo "sum of is errors squared: $varError number of failure: $numFailures ($percentFailures %)"

