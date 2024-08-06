import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import json

from inference_model import HumanHulk


def save_output(outputs, metadata_path):
    with open(metadata_path, 'w') as f:
        json.dump(outputs['metadata'], f)
    if outputs['images']['kp_img'] is not None:
        outputs['images']['kp_img'].save(outputs['metadata']['kp_path'])
    if outputs['images']['CIHP_img'] is not None:
        outputs['images']['CIHP_img'].save(outputs['metadata']['CIHP_path'])


def multi_inference(rank, world_size):
    setup(rank, world_size)

    device = torch.device(rank)
    pipeline = HumanHulk(device)
    img_path = 'your-image-path'
    output = dict(metadata=dict(), images=dict())
    output_path = 'metadata_path'
    output['metadata']['box'] = pipeline.set_image(img_path)
    if output['metadata']['box'] is not None:
        output['metadata']['CIHP_path'] = 'cihp_img_path'
        output['metadata']['kp_path'] = 'keypoint_img_path'
        output['metadata']['keypoints'], output['images']['kp_img'] = pipeline.get_pose(img_path)
        output['images']['CIHP_img'] = pipeline.get_parse()
    else:
        output['metadata']['CIHP_path'] = None
        output['metadata']['kp_path'] = None
        output['metadata']['keypoints'] = None
        output['images']['kp_img'] = None
        output['images']['CIHP_img'] = None
    save_output(output, output_path)

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
