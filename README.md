# SR

srun --ntasks=1 --partition ALL --account rpixel --qos premium --cpus-per-task=1 --mem=10G --gres=gpu:1 --time=72:00:00 --pty bash

sacct -u zeyul --format=User,JobID,Jobname,partition,state,time,start,end,elapsed,MaxRss,MaxVMSize,nnodes,ncpus,nodelist