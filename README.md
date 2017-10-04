# Photographic Image Synthesis with Cascaded Refinement Networks

This is a Tensorflow implementation of cascaded refinement networks to synthesize photographic images from semantic layouts.

<img src="http://cqf.io/ImageSynthesis/teaser.png"/>

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.0) + Scipy + Numpy + Pillow.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.

### Quick Start (Testing)
1. Clone this repository.
2. Download the pretrained models from Google Drive by running "python download_models.py". It takes several minutes to download all the models.
3. Run "python demo_512p.py" or "python demo_1024p.py" (requires large GPU memory) to synthesize images.
4. The synthesized images are saved in "result_512p/final" or "result_1024p/final".

### Training
To train a model at 256p resolution, please set "is_training=True" and change the file paths for training and test sets accordingly in "demo_256p.py". Then run "demo_256p.py".

To train a model at 512p resolution, we fine-tune the pretrained model at 256p using "demo_512p.py". Also change "is_training=True" and file paths accordingly.

To train a model at 1024p resolution, we fine-tune the pretrained model at 512p using "demo_1024p.py". Also change "is_training=True" and file paths accordingly.

## Video
https://youtu.be/0fhUJT21-bs

## Citation
If you use our code for research, please cite our paper:

Qifeng Chen and Vladlen Koltun. Photographic Image Synthesis with Cascaded Refinement Networks. In ICCV 2017.

## Amazon Turk Scripts
The scripts are put in the folder "mturk_scripts".

## Todo List
1. Add the code and models for the GTA dataset.

## Question
If you have any question or request about the code and data, please email me at chenqifeng22@gmail.com. If you need the pretrained model on NYU, please send an email to me.

## License
MIT License
