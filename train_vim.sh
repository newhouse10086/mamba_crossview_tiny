#!/bin/bash

# VIM模型训练配置
name="FSRA_VIM"
backbone="vim_small"
pretrain_path="vim_t_midclstok_ft_78p3acc.pth"

# 数据路径 - 请根据你的实际路径修改
data_dir="/path/to/your/University1652/train"
test_dir="/path/to/your/University1652/test"

# 训练参数
gpu_ids=0
num_worker=4
lr=0.01
sample_num=1
block=3
batchsize=8
triplet_loss=0.3
num_epochs=120
pad=0
views=2

echo "开始使用VIM模型训练FSRA..."
echo "模型名称: $name"
echo "Backbone: $backbone"
echo "预训练权重: $pretrain_path"

# 检查预训练权重文件是否存在
if [ ! -f "$pretrain_path" ]; then
    echo "错误: 找不到预训练权重文件 $pretrain_path"
    echo "请确保vim_t_midclstok_ft_78p3acc.pth在当前目录下"
    exit 1
fi

# 训练命令
python train.py \
  --name $name \
  --backbone $backbone \
  --pretrain_path $pretrain_path \
  --data_dir $data_dir \
  --gpu_ids $gpu_ids \
  --num_worker $num_worker \
  --views $views \
  --lr $lr \
  --sample_num $sample_num \
  --block $block \
  --batchsize $batchsize \
  --triplet_loss $triplet_loss \
  --num_epochs $num_epochs

echo "训练完成！"

# 测试阶段
echo "开始测试..."
cd checkpoints/$name

for((i=119;i<=$num_epochs;i+=10));
do
  for((p = 0;p<=$pad;p+=10));
  do
    for ((j = 1; j < 3; j++));
    do
        python ../../test_server.py --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker --pad $pad
    done
  done
done

echo "所有任务完成！" 