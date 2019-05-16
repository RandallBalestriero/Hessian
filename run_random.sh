#
screen -dmS randcnna bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_random.py --model cnn --dataset cifar10 --proportion 0";
screen -dmS randcnnb bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_random.py --model cnn --dataset cifar10 --proportion 0.25";
screen -dmS randcnnc bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_random.py --model cnn --dataset cifar10 --proportion 0.5";
screen -dmS randcnnd bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_random.py --model cnn --dataset cifar10 --proportion 0.75";



