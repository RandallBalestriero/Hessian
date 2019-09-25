#
GPU=5

#for epsilon in 0.1 0.001 0.0001
#do
for model in cnn simpleresnet
do
    for augmentation in 0 1
    do
        screen -dmS Jmnist$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python run_J.py --model $model --dataset cifar10 --augmentation $augmentation --lr 0.0005";
        GPU=$(((GPU+1)%8));
        screen -dmS Jsvhn$model bash -c "export CUDA_VISIBLE_DEVICES=$GPU;python run_J.py --model $model --dataset svhn --augmentation $augmentation --lr 0.0005";
        GPU=$(((GPU+1)%8));
    done
done
#done







