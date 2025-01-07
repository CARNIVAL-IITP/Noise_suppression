# CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
rm -rf ./wandb/
rm -rf ./__pycache__
rm -rf ./*/__pycache__
rm -rf ./*/*/__pycache__

rm -rf ../results

exp="../exp_results/$(date '+%Y_%m_%d_%H_%M_%S')_dccrn_SPP_kd_cosin_max100"
mkdir -p $exp
cp -r "../src" $exp
cd $exp/src
rm -rf ./.git
mkdir ../results

echo $exp


device=0
echo "device: $device"

python train.py 
python inference.py 