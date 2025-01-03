import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/dataset/imagenet1k/data",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )

    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training (default value same as args.seed)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default="./results",
        type=str,
    )
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")
    parser.add_argument("--use_wandb", action="store_true", help="use weights and biases")
    parser.add_argument("--wandb_name", type=str, default=None, help="name of wandb")
    parser.add_argument("--evaluate_knn", action="store_true", help="evaluate knn during unlearning")
    parser.add_argument("--evaluate_cka", action="store_true", help="evaluate cka during unlearning")
    parser.add_argument("--retrained_model_path", type=str, default="./pretrained_model/retrained_random100.pth.tar", help="the path of retrained model") # set your retrained model path
    parser.add_argument("--office_home_dataset_path", type=str, default="/home/dataset/office-home", help="the path of office-home dataset") # set your office-home dataset path
    parser.add_argument("--cub_dataset_path", type=str, default="/home/dataset/CUB/CUB_200_2011/images", help="the path of cub dataset") # set your cub dataset path
    parser.add_argument("--domainnet_dataset_path", type=str, default="/home/dataset/domainnet", help="the path of domainnet dataset") # set your domainnet dataset path
    parser.add_argument("--original_Df", action="store_true", default=False, help="use not transformed Df")
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        choices=["retain", "forget", None],
        help="Type of data to use for CKA analysis",
    )
    parser.add_argument(
        "--aug_method", 
        type=str, 
        nargs='+',
        default=['original', 'gaussian', 'crop_resize', 'color_distortion', 
                'color_jitter', 'rotation', 'cutout', 'gaussian_blur', 
                'mixup', 'cutmix'],
        help="augmentation methods to use"
    )

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=1000, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind epochs")

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )
    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace", 
        default="random100",
        choices=["random100", "random200", "top100_officehome_real", "top100_cub", "top100_domainnet", "top200_officehome_real", "top200_cub", "top200_domainnet"],
        help="Specific class to forget"
    )
    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")
    parser.add_argument("--mask_path", default=None, type=str, help="the path of saliency map")

    return parser.parse_args()
