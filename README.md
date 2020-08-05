# GCN

## Dependencies
  * Tensorflow
  * Pandas
  * Numpy
  * Sklearn
  * tqdm

## Workflow
1. Convert ABIDE_fc.mat to csv files so that data become easily readable to Python. Run
```
converter.m
```
in MatLab

2. Use data.py to generate a pickle file which contains the training, validation, test datasets. This is to make sure the split of datasets is the same across multiple runs of train.py because the data is shuffled before being split.<br>
(Data paths are specified in config.py: <br>
DATA_dir, left_table_file, matrices_dir, pickle_path, upsampled_pickle_path)
```
$ python data.py
```

3. Train model. The datasets are read from [pickle_path] or [upsampled_pickle_path], as specified in config.py
```
$ python train.py
```
Or save model to a specified directory under [ckpt_dir]/[model_idx] where [ckpt_dir] is specified in config.py
```
$ python train.py [model_idx]
```
Or use select_model.sh and run 50 times the above command
```
./select_model.sh
```

4. Evaluate model. This evaluates the model saved to [ckpt_dir]
```
$ python eval.py
```
Or use Tensorboard, go to the code root directory and run
```
tensorboard --logdir logs
```

## Note
1. The original data file: ABIDE_fc.mat
2. Model is defined in: model.py
3. When optimized on test acc, the current model can reach an acc of ~0.6968
