#
screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_hessian.py --model resnetlarge --dataset cifar10 --data_augmentation 0 --regularization 0";
screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_hessian.py --model resnetlarge --dataset cifar10 --data_augmentation 1 --regularization 0";

screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=1;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 0 --regularization 0";
screen -dmS d bash -c "export CUDA_VISIBLE_DEVICES=1;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 1 --regularization 0";




screen -dmS e bash -c "export CUDA_VISIBLE_DEVICES=2;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 0 --regularization 0.01";
screen -dmS f bash -c "export CUDA_VISIBLE_DEVICES=3;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 0 --regularization 0.1";
screen -dmS g bash -c "export CUDA_VISIBLE_DEVICES=4;
python quickstart_hessian.py --model cnn --dataset cifar10 --data_augmentation 0 --regularization 1";


screen -dmS h bash -c "export CUDA_VISIBLE_DEVICES=5;
python quickstart_hessian.py --model resnetlarge --dataset cifar10 --data_augmentation 0 --regularization 0.01";
screen -dmS i bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_hessian.py --model resnetlarge --dataset cifar10 --data_augmentation 0 --regularization 0.1";
screen -dmS j bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_hessian.py --model resnetlarge --dataset cifar10 --data_augmentation 0 --regularization 1";






