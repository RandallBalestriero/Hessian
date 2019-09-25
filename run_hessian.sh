screen -dmS e bash -c "export CUDA_VISIBLE_DEVICES=3;
python quickstart_hessian.py --model cnn --dataset cifar100 --data_augmentation 0 --regularization 0";

screen -dmS f bash -c "export CUDA_VISIBLE_DEVICES=4;
python quickstart_hessian.py --model cnn --dataset cifar100 --data_augmentation 1 --regularization 0";

screen -dmS g bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_hessian.py --model resnetlarge --dataset cifar100 --data_augmentation 0 --regularization 0";

screen -dmS h bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_hessian.py --model resnetlarge --dataset cifar100 --data_augmentation 1 --regularization 0";








