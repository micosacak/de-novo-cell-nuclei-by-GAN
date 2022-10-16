#!/bin/bash
#SBATCH --ntasks=24
#SBATCH --mem=512gb
#SBATCH --time=9999999:05:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=mask2dapi_%j.out
#SBATCH --job-name=mask2dapi
virtual_env=""
working_dir="~"
path2traindata="~"
source $virtual_env
cd working_dir
for model in mask2dapi_002a mask2dapi_002b mask2dapi_003a mask2dapi_003b
do for kernel_size in 3 5 7
        python masks2dapi_training.py -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
done
done

model="mask2dapi_001"
for filter_size in 16 32
do for kernel_size in 3 5 7
        python masks2dapi_training.py -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
done
done

model="mask2dapi_001"
for filter_size in 64
do for kernel_size in 3 5
        python masks2dapi_training.py -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 10 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
        python masks2dapi_training.py -l -r -R 1 -b 20 -k $kernel_size -i 128 -f $filter_size -g 0 -m $model -P $path2traindata
done
done


