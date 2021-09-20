# Usage of image classifier

The data is set under the `data` folder, and the source code is set under `src` folder. 

Run the code as follows:

```
cd src
python3 run.py \
    -d ../data \
    -lr 0.001 \
    --read-mode gray \
    --batch-size 64 \
    --seed 1234 \
    --num-cnn 2 \
    --num-fc 3 \
    --epoch 10 \
    --class-num 2 \
    --val-data 0.2 \
    --ok-weight 0.2 \
    --binary-image \
    --ext \
    -o ../output/
```
The arguments are described as follows:

```
  -h, --help            show this help message and exit
  --data-path DATA_PATH, -d DATA_PATH
                        The path to train/test data files.
  --read-mode {gray,rgba,rgb}, -rm {gray,rgba,rgb}
                        Select a mode to read in images.
  --binary-image, -bi   Binarize image
                        Binarize input images.
  --val-data VAL_DATA   Choose the proportion of val data (from 0 to 1).
  --reshape, -rs        Reshape image to 256x256
  --ext-data            Extend data by rotating and fliping
  --random-rotate       Random rotate in training
  --learning-rate <float>, -lr <float>
                        Learning rate
  --batch-size <int>    Batch size
  --ok-weight <float>   Weight of loss for the class of *ok*
  --seed <int>          Random seed
  --num-cnn <int>       Number of CNN layers.
  --num-fc <int>        Number of FC layer.{1, 3}
  --epoch <int>         Number epochs
  --class-num <int>     Number of classes
  --output-path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to save results
```

An example to run the code is given in `src/run.sh`.
