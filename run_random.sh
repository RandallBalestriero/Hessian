#
screen -dmS randcnna bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_random.py --model mlpsmall --dataset cifar10 --proportion 0";
screen -dmS randcnnb bash -c "export CUDA_VISIBLE_DEVICES=0;
python quickstart_random.py --model mlpsmall --dataset cifar10 --proportion 0.25";
screen -dmS randcnnc bash -c "export CUDA_VISIBLE_DEVICES=2;
python quickstart_random.py --model mlpsmall --dataset cifar10 --proportion 0.5";
screen -dmS randcnnd bash -c "export CUDA_VISIBLE_DEVICES=2;
python quickstart_random.py --model mlpsmall --dataset cifar10 --proportion 0.75";

#
screen -dmS randcnna bash -c "export CUDA_VISIBLE_DEVICES=3;
python quickstart_random.py --model mlplarge --dataset cifar10 --proportion 0";
screen -dmS randcnnb bash -c "export CUDA_VISIBLE_DEVICES=4;
python quickstart_random.py --model mlplarge --dataset cifar10 --proportion 0.25";
screen -dmS randcnnc bash -c "export CUDA_VISIBLE_DEVICES=5;
python quickstart_random.py --model mlplarge --dataset cifar10 --proportion 0.5";
screen -dmS randcnnd bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_random.py --model mlplarge --dataset cifar10 --proportion 0.75";





