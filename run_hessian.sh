#
screen -dmS cnna bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 0";
screen -dmS cnnb bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 1";

screen -dmS cnna bash -c "export CUDA_VISIBLE_DEVICES=1;
python quickstart_hessian.py --model cnn --dataset svhn --data_augmentation 0";
screen -dmS cnnb bash -c "export CUDA_VISIBLE_DEVICES=1;
python quickstart_hessian.py --model cnn --dataset svhn --data_augmentation 1";

screen -dmS cnnc bash -c "export CUDA_VISIBLE_DEVICES=2;
python quickstart_hessian.py --model cnn --dataset mnist --data_augmentation 0";
screen -dmS cnnd bash -c "export CUDA_VISIBLE_DEVICES=2;
python quickstart_hessian.py --model cnn --dataset mnist --data_augmentation 1";

#
screen -dmS cnne bash -c "export CUDA_VISIBLE_DEVICES=3;
python quickstart_hessian.py --model resnetsmall --dataset cifar10 --data_augmentation 0";
screen -dmS cnnf bash -c "export CUDA_VISIBLE_DEVICES=3;
python quickstart_hessian.py --model resnetsmall --dataset cifar10 --data_augmentation 1";

screen -dmS cnng bash -c "export CUDA_VISIBLE_DEVICES=4;
python quickstart_hessian.py --model resnetsmall --dataset svhn --data_augmentation 0";
screen -dmS cnnh bash -c "export CUDA_VISIBLE_DEVICES=4;
python quickstart_hessian.py --model resnetsmall --dataset svhn --data_augmentation 1";

screen -dmS cnni bash -c "export CUDA_VISIBLE_DEVICES=5;
python quickstart_hessian.py --model resnetsmall --dataset mnist --data_augmentation 0";
screen -dmS cnnh bash -c "export CUDA_VISIBLE_DEVICES=5;
python quickstart_hessian.py --model resnetsmall --dataset mnist --data_augmentation 1";





