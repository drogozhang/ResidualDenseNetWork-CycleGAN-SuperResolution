# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-01

isTrain = True
gpu_ids = '0'

continue_train = False
which_epoch = -1  # if continue Train ture, set this value

print_net_in_detail = False
lr_decay_iters = 500
dataDir = './DIV2K'
saveDir = './result'
checkpoints_dir = './checkpoints_dir'
LR_rate = '4X'  # '2X'    # tuning here!
load = 'RDN_' + LR_rate
model_name = 'RDN'
need_patch = True

downsample_method = 'conv'  # 'pooling'
nDenselayer = 3  # 6
growthRate = 32
nBlock = 16
nFeat = 64
input_channel = 3
patchSize = 144

nThreads = 4
batchSize = 16
lr = 1e-4
beta = 0.5
epochs = 10000
lrDecay = 2000
decayType = 'step'
lossType = 'L1'  # 'MSE'
scale = int(LR_rate[0])  # 1

pool_size = 50

lambda_A = 10
lambda_B = 10
lambda_identity = 0.5
which_direction = 'AtoB'

print_freq = batchSize * 5
save_epoch_freq = 5

# --model_name RDN --load demo_x3_RDN --dataDir .// --need_patch   --patchSize  --nDenselayer  --nFeat 64 --growthRate 32
# --scale 3 --epoch 10000 --lrDecay 2000 --lr 1e-4 --batchSize 16 --nThreads 4 --lossType L1
