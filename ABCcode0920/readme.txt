Code for the paper entitled "ABC: Auxiliary Balanced Classifier for Class-imbalanced Semi-supervised Learning"
-------------------------------------------------------------------------------------------------------------------------------------------------------
Dependencies

python3.7

torch 1.5.1 (python3.7 -m pip install torch==1.5.1)
torchvision 0.6.1 (python3,7 -m pip install torchvision==0.6.1)
numpy 1.19,4 (python3.7 -m pip install numpy==1.19.4)
scipy (python3.7 -m pip install scipy)
randAugment (python3.7 -m pip install git+https://github.com/ildoonet/pytorch-randaugment), (if an error occurs, type apt-get install git)
tensorboardX (python3.7 -m pip install tensorboadX)
matplotlib (python3.7 -m pip install matplotlib)
progress (python3.7 -m pip install progress)

-------------------------------------------------------------------------------------------------------------------------------------------------------

if you want to run ABCremix.py with 0th gpu , ratio of labeled data as 20%, 
N1(number of data points belonging to first class = num_max) as 1000 , ratio of imbalance as 100, 500 epoch with each epoch 500 iteration, manualseed as 0, dataset as CIFAR-10, imbalance type with long tailed imbalance:

python3.7 ABCremix.py --gpu 0 --label_ratio 20 --num_max 1000 --imb_ratio 100 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar10 --imbalancetype long

if you want to run ABCfix.py with 0th gpu , ratio of labeled data as 40%, 
N1(number of data points belonging to first class = num_max) as 200 , ratio of imbalance as 20, 500 epoch with each epoch 500 iteration, manualseed as 0, dataset as CIFAR-100, imbalance type with step imbalance:

python3.7 ABCfix.py --gpu 0 --label_ratio 40 --num_max 200 --imb_ratio 20 --epoch 500 --val-iteration 500 --manualSeed 0 --dataset cifar100 --imbalancetype step

-------------------------------------------------------------------------------------------------------------------------------------------------------

These codes validate peformance of algorithms on testset after each epoch of training

-------------------------------------------------------------------------------------------------------------------------------------------------------

Performance of algorithms are summarized in Section 4 of the paper
