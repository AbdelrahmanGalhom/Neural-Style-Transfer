import argparse
import os
import sys
import torch
from neural_style.neural_style import check_paths, train, stylize


def build_parser():
    """Build command line parser for the Neural Style Transfer application"""
    main_parser = argparse.ArgumentParser(description="Neural Style Transfer CLI")
    subparsers = main_parser.add_subparsers(title="commands", dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new Neural Style Transfer model")
    train_parser.add_argument("--epochs", type=int, default=2,
                             help="number of training epochs, default is 2")
    train_parser.add_argument("--batch-size", type=int, default=4,
                             help="batch size for training, default is 4")
    train_parser.add_argument("--dataset", type=str, required=True,
                             help="path to training dataset, the path should point to a folder "
                                  "containing another folder with all the training images")
    train_parser.add_argument("--style-image", type=str, required=True,
                             help="path to style-image")
    train_parser.add_argument("--save-model-dir", type=str, required=True,
                             help="path to folder where trained model will be saved.")
    train_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                             help="path to folder where checkpoints of trained models will be saved")
    train_parser.add_argument("--image-size", type=int, default=256,
                             help="size of training images, default is 256 X 256")
    train_parser.add_argument("--style-size", type=int, default=None,
                             help="size of style-image, default is the original size of style image")
    train_parser.add_argument('--accel', action='store_true',
                             help='use accelerator')
    train_parser.add_argument("--seed", type=int, default=42,
                             help="random seed for training")
    train_parser.add_argument("--content-weight", type=float, default=1e5,
                             help="weight for content-loss, default is 1e5")
    train_parser.add_argument("--style-weight", type=float, default=1e10,
                             help="weight for style-loss, default is 1e10")
    train_parser.add_argument("--lr", type=float, default=1e-3,
                             help="learning rate, default is 1e-3")
    train_parser.add_argument("--log-interval", type=int, default=500,
                             help="number of images after which the training loss is logged, default is 500")
    train_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                             help="number of batches after which a checkpoint of the trained model will be created")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Use a trained model to stylize images")
    eval_parser.add_argument("--content-image", type=str, required=True,
                            help="path to content image you want to stylize")
    eval_parser.add_argument("--content-scale", type=float, default=None,
                            help="factor for scaling down the content image")
    eval_parser.add_argument("--output-image", type=str, required=True,
                            help="path for saving the output image")
    eval_parser.add_argument("--model", type=str, required=True,
                            help="saved model to be used for stylizing the image.")
    eval_parser.add_argument('--accel', action='store_true',
                            help='use accelerator')

    return main_parser


def main():
    """Process command line arguments and run neural style transfer"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Check for accelerator availability
    if args.accel and not torch.accelerator.is_available():
        print("ERROR: accelerator is not available, try running on CPU")
        sys.exit(1)
    if not args.accel and torch.accelerator.is_available():
        print("WARNING: accelerator is available, run with --accel to enable it")

    # Run the appropriate function based on command
    if args.command == "train":
        check_paths(args)
        train(args)
    elif args.command == "eval":
        stylize(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
