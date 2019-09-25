#
N=10000
GPU=5

#for epsilon in 0.1 0.001 0.0001
#do
for gamma in 0.0001 0.00001
do
#    screen -dmS hhessian$gamma bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i run.py --model simpleresnet --dataset cifar10 -n $N --gamma $gamma --lr 0.0001";
#    GPU=$(((GPU+1)%8));
    screen -dmS hhessian2$gamma bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python -i run.py --model cnn --dataset cifar10 -n $N --gamma $gamma --lr 0.0001";
    GPU=$(((GPU+2)%8));
done
#done







