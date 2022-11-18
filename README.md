# CWGAN-GP
Conditioning Wasserstein Generative Adversarial Network with Gradient Penalty for Geological Modeling.


This is a CWGAN-GP for geological modeling process.

Before running the code, some packages should be installed:

Python 3.9.5
PyTorch 1.10.1
opencv-python 4.5.5
numpy 1.21.2
scipy 1.7.3
matplotlib 3.5.0


For 2D CWGAN-GP:
Some directories are described as follows:
Conditional information: for the testing datset, save the conditioning information with both sgems format and the gray-picture, saved in the conditional_data directories
and conditional_picture respectively.

fake_images: save the real images and corresponding generation results as jpeg format when the validation process is conducted.

test_final_results: save the generation results as sgems format when the validation process is conducted.

test_real_data: save the testing datasets as sgems format.

Training_image_Set: save the 2D subsurface models, including 34596 2D pictures with size 64*64.

WGAN_GP_Model: save the generative models during training.

For running the code, firstly, you should run the Training_Image.py to prepare the datasets, then you can open mainWGAN.py and trainingWGAN.py respectively so that
different parameters could be set, then run the mainWGAN.py to train the model.
For validation, you can open the mainWGAN.py, finding the validation code and cancel the upper training code, choose the saved the model then the validation could be conducted.



For 3D CWGAN-GP:




