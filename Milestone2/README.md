# Project code template

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vita-epfl/introML-2021/blob/main/project/train.ipynb)

This is a PyTorch code template for milestone 2 of the project, which consists of identifying whether buildings have been washed away by a tsunami from aerial images.

This code contains everything needed to load the dataset, view image pairs, train a model and generate a CSV submission file with predictions.

## Dependencies
All required packages can be found in `requirements.txt`.

## Project structure

### Data

**You can find the dataset [here](https://drive.google.com/file/d/1otKxIvEP77Cap9VmUkujMrAMo4K8_F1c/view?usp=sharing).**

This project uses the fixed scale images from the [AIST Building Change Detection dataset](https://github.com/gistairc/ABCDdataset), which consists of pairs of pre- and post-tsunami aerial images. These images should be placed in a directory named `patch-pairs` inside the `data` directory.  
We also provide two CSV files:
- `train.csv` which contains the path to each image in the training set, as well as the target (0 for "surviving", 1 for "washed-away").
- `test.csv` which contains the path to each image in the test set.

### Code

The notebook `train.ipynb` contains a complete training procedure. Feel free to modify it as desired to improve the performance of your model.


In addition, here is a brief description of what each of the provided Python files does:
- `dataset.py`: Contains `PatchPairsDataset`, a PyTorch Dataset class that loads pairs of images and their target, as well as a function to split datasets into training and validation sets.
- `evaluator.py`: Evaluates and generates prediction from a trained model
- `metrics.py`: Metrics to keep track of the loss and accuracy during training
- `trainer.py`: Contains `Trainer`, a class which implements the training loop as well as utilities to log the training process to TensorBoard, and load & save models
- `utils.py`: Utilities for displaying pairs of images and generating a submission CSV

You are free to modify all of these Python files as desired for your experiments, although this is not necessary to achieve a very good performance in this challenge. If you are using Google Colab, keep in mind that any changes to files besides `train.ipynb` will get discarded when your session terminates.

## Experiment logging
By default, all runs are logged using [TensorBoard](https://www.tensorflow.org/tensorboard), which keeps track of the loss and accuracy. 
After installing TensorBoard, type
```
tensorboard --logdir=runs
```
in the terminal to launch it.

Alternatively, TensorBoard can be launched directly from notebooks, refer to `train.ipynb` for more info.

For more information on how to use TensorBoard with PyTorch, check out [the documentation](https://pytorch.org/docs/stable/tensorboard.html).

## Google Colab

You can run this notebook in Colab using the following link: https://colab.research.google.com/github/vita-epfl/introML-2021/blob/main/project/train.ipynb

**Important info:** 
- To train models much quicker, switch to a GPU runtime (*Runtime -> Change runtime type -> GPU*)
- Copy the Colab notebook to your Google Drive (*File -> Save a copy in Drive*) so that your changes to the training notebook persist.
- All files with the exception of the training notebook (`train.ipynb`) get deleted when your session terminates. Make sure to download all the relevant files (e.g. submissions, trained models, logs) before ending your session.
- It is not possible for multiple people to edit a Colab notebook at the same time. To avoid this issue, wait for your teammate to terminate their session before beginning to work on the notebook, or work on a copy and combine your changes afterwards.
