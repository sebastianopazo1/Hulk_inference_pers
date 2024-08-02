import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp

from inference_pipe import HumanHulk

def multi_inference(rank, world_size):    
    setup(rank, world_size)    

    device = torch.device(rank)
    pipeline = HumanHulk(device)
    img_path = 'your-image-path'
    box = pipeline.set_image(img_path)
    keypoints, pose_img = pipeline.get_pose(img_path)
    parse_img = pipeline.get_parse()

    cleanup()
    
    
def cleanup():
    dist.destroy_process_group()
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def run(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)   
  
    
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    run(multi_inference, world_size)
