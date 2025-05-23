# Neural Style Transfer

This repository contains a modular PyTorch implementation of an algorithm for artistic style transfer. The application provides an easy-to-use command-line interface through the main.py script. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

**Live Demo**: Try this model on our [Hugging Face Space](https://huggingface.co/spaces/AbdelrahmanGalhom/Style-Transfer)

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Requirements

The program is written in Python, and uses [PyTorch](http://pytorch.org/) and [SciPy](https://www.scipy.org). You can install the required packages using:

```bash
pip install -r requirements.txt
```

A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using pre-trained models.

## Usage

Stylize an image

```
python main.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --accel
```

- `--content-image`: path to content image you want to stylize.
- `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
- `--output-image`: path for saving the output image.
- `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
- `--accel`: use accelerator if available (CUDA, MPS, XPU)

Train a new style model

```bash
python main.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --accel
```

- `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. COCO 2014 Training images dataset [80K/13GB] [(download)](https://cocodataset.org/#download) is recommended.
- `--style-image`: path to style-image.
- `--save-model-dir`: path to folder where trained model will be saved.
- `--epochs`: number of training epochs, default is 2.
- `--accel`: use accelerator if available.

## Project Structure

This project has been organized with a clean, modular structure:

- `main.py` - Main entry point for the application
- `neural_style/` - Core implementation module
  - `neural_style.py` - Contains the train and stylize functions
  - `transformer_net.py` - The neural network architecture for style transfer
  - `vgg.py` - VGG16 implementation used for feature extraction
  - `utils.py` - Utility functions for image loading/saving and other operations
- `images/` - Example images for testing
  - `content-images/` - Sample images to stylize
  - `style-images/` - Sample style images
  - `output-images/` - Example output images
- `saved_models/` - Pre-trained style models

## Additional Information

### Advanced Parameters

For advanced users, you can fine-tune these parameters during training:

- `--content-weight`: controls the influence of content image (default: 1e5)
- `--style-weight`: controls the influence of style image (default: 1e10)
- `--lr`: learning rate for the optimizer (default: 1e-3)
- `--image-size`: size of training images (default: 256 x 256)
- `--batch-size`: batch size for training (default: 4)
- `--seed`: random seed for reproducible results (default: 42)
- `--checkpoint-interval`: batches between checkpoint saves (default: 2000)
- `--log-interval`: images after which training loss is logged (default: 500)

### Training Tips

For training new models, you might need to adjust `--content-weight` and `--style-weight`. The mosaic style model shown above was trained with `--content-weight 1e5` and `--style-weight 1e10`. Other models used similar parameters with slight variations in the `--style-weight` (`5e10` or `1e11`).

For best results:
1. Use a diverse dataset of content images
2. Train for at least 2 epochs (more for complex styles)
3. Use hardware acceleration (`--accel`) if available
4. Consider starting with pre-trained weights and fine-tuning

### Pre-trained Models

The repository includes several pre-trained models in the `saved_models` directory:
- `mosaic.pth`: Mosaic style based on a stained glass painting
- `candy.pth`: Candy style with vibrant colors
- `rain_princess.pth`: Moody blue rain princess style
- `starry_night.pth`: Style based on Van Gogh's Starry Night
- `udnie.pth`: Abstract udnie style

## Getting Started

To get started with Neural Style Transfer:

1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`
3. Run the style transfer with the included pre-trained models:
   ```
   python main.py eval --content-image images/content-images/amber.jpg --model saved_models/mosaic.pth --output-image output.jpg
   ```

Or try the [interactive demo](https://huggingface.co/spaces/AbdelrahmanGalhom/Style-Transfer) on Hugging Face!
