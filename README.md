# Ensemble-Learning-for-Face-classification
Please refer to the provided pdf file for more information.

This project is for Kaggle competition:https://www.kaggle.com/c/11785-f19-hw2p2-classification/overview

## Download
Download the data set from https://www.kaggle.com/c/11785-f19-hw2p2-classification/overview

## Train three networks from scatch
Train ShuffleNet, MobileNet V2, ResNet34 from scratch. These models are slightly different from the officail models. This is mainly because the input image size is 32x32 (much smaller). Models are provided in MobileNetV2_2.py and ResNet.ipynb

## Make the embeddings for each input image and train a Multilayer Perceptron on those embeddings
Using embedding.ipynb to make embedding for train data. Use embedding_network.ipynb to train an MLP on the training data. Use embedding_test_xxx.ipynb to make embeddings for test images. Use kaggle_test_result.ipynb to get final test results. Submit the results on the provided kaggle competition to get the final test accuracy.


