import torch
import torch.nn as nn
from core.models.model_entry import aio_entry_v2mae_shareneck
from PIL import Image
from core.utils import NestedTensor
import core.models.decoders as decoders
import core.models.backbones as backbones
import core.models.necks as necks
import core.models.output_projector as output_projector
import core.models.input_adapter as input_adapter
from easydict import EasyDict as edict
from core.config_inference import Config_Hulk
from dict_recursive_update import recursive_update
import yaml
import re
import numpy as np
from draw_utils import draw_pose_from_cords, mmpose_to_coco
import torch.nn.functional as F
from torchvision import transforms

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def count_parameters_num(model):
    count = 0
    count_fc = 0
    param_dict = {name: param for name, param in model.named_parameters()}
    param_keys = param_dict.keys()
    for m_name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count += temp_params.data.nelement()
        elif isinstance(m, nn.Linear):
            weight_name = m_name + '.weight'
            bias_name = m_name + '.bias'
            if weight_name in param_keys:
                temp_params = param_dict[weight_name]
                count_fc += temp_params.data.nelement()
            if bias_name in param_keys:
                temp_params = param_dict[bias_name]
                count_fc += temp_params.data.nelement()
    print('Number of conv/bn params: %.2fM' % (count / 1e6))
    print('Number of linear params: %.2fM' % (count_fc / 1e6))


def create_model(config, device):
    patch_adapter_module = input_adapter.patchembed_entry(config.patch_adapter)
    label_adapter_module = input_adapter.patchembed_entry(config.label_adapter)

    # build backbone
    backbone_module = backbones.backbone_entry(config.backbone)
    count_parameters_num(backbone_module)
    config.patch_neck.kwargs.backbone = backbone_module
    patch_neck_module = necks.neck_entry(config.patch_neck)
    config.label_neck.kwargs.backbone = backbone_module
    label_neck_module = necks.neck_entry(config.label_neck)

    # build decoder(s)
    config.decoder.kwargs.backbone = backbone_module
    config.decoder.kwargs.neck = patch_neck_module
    config.decoder.kwargs.patch_adapter = patch_adapter_module
    config.decoder.kwargs.label_adapter = label_adapter_module
    config.decoder.kwargs.patch_neck = patch_neck_module
    config.decoder.kwargs.label_neck = label_neck_module
    # dataset = datasets.dataset_entry(config.dataset)

    if config.dataset.type == "COCOStuffSegDatasetDev":
        config.decoder.kwargs.ignore_value = config.dataset.kwargs.cfg.ignore_value
        config.decoder.kwargs.num_classes = config.dataset.kwargs.cfg.num_classes
    elif config.dataset.type in ["COCOPosDatasetDev", "MultiPoseDatasetDev", 'MPIIPosDatasetDev']:
        num_classes = 2  # COCO: ['person'] + ['__background__']
        config.decoder.kwargs.num_classes = num_classes if config.dataset.type != 'MPIIPosDatasetDev' else 16
        config.decoder.kwargs.ignore_value = None
    elif "ParsingDataset" in config.dataset.type:
        config.decoder.kwargs.ignore_value = config.dataset.kwargs.cfg.ignore_value
        config.decoder.kwargs.num_classes = config.dataset.kwargs.cfg.num_classes
    elif config.dataset.type in ['MultiAttrDataset', 'mmSkeletonDataset']:
        config.decoder.kwargs.ignore_value = None
        # compatablity fix, will be removed, not effective
        config.decoder.kwargs.num_classes = 0
    elif config.dataset.type in ["PedestrainDetectionDataset_v2", 'CrowdHumanDetDataset', "PedestrainDetectionDataset_v2demo"]:
        config.decoder.kwargs.ignore_value = None
        # treat pedestrain classificatin as a binary classification
        config.decoder.kwargs.num_classes = 1
    elif config.dataset.type in ['CocoCaption', 'CocoCaptiondemo']:
        config.decoder.kwargs.ignore_value = None
        config.decoder.kwargs.num_classes = 1
    elif config.dataset.type in ["MeshTSVYamlDataset"]:
        config.decoder.kwargs.ignore_value = None
        config.decoder.kwargs.num_classes = 1  # No class required
    else:
        raise NotImplementedError

    config.decoder.kwargs.ginfo = None
    config.decoder.kwargs.bn_group = None
    decoder_module = decoders.decoder_entry(config.decoder)

    # build output project using the setting of corresponding input adapters
    patch_proj_kwargs_dict = {'kwargs': {'hidden_dim': config.decoder.kwargs.transformer_predictor_cfg.hidden_dim,
                                         'patch_size': patch_adapter_module.patch_size,
                                         'in_chans': patch_adapter_module.in_chans,
                                         'stride_level': patch_adapter_module.stride_level, }
                              }
    patch_proj_loss_cfg_kwargs_dict = {'kwargs': {
        'patch_size': patch_adapter_module.patch_size[0],
        'stride': patch_adapter_module.stride_level,
    }}

    # rgb branch has a default kwargs - extra_norm_pix_loss,
    # use recursive_update to update other kwargs.
    recursive_update(config.patch_proj, patch_proj_kwargs_dict)
    recursive_update(config.patch_proj.kwargs.loss_cfg,
                     patch_proj_loss_cfg_kwargs_dict)
    patch_proj_module = output_projector.outputproj_entry(config.patch_proj)

    label_proj_kwargs_dict = {
        'kwargs': {'hidden_dim': config.decoder.kwargs.transformer_predictor_cfg.hidden_dim,
                   'patch_size': label_adapter_module.patch_size,
                   'in_chans': label_adapter_module.in_chans,
                   'stride_level': label_adapter_module.stride_level,
                   'loss_cfg':
                   {'kwargs':
                    {'patch_size': label_adapter_module.patch_size[0],
                     'stride': label_adapter_module.stride_level,
                     }},
                   }
    }

    recursive_update(config.label_proj, label_proj_kwargs_dict)
    label_proj_module = output_projector.outputproj_entry(config.label_proj)

    modalities = {
        'patch': config.patch_adapter.type.split('_adapter')[0],
        'label': config.label_adapter.type.replace('_adapter', ''),
    }
    is_training = config.get('is_training', False)
    backbone_module.training = \
        patch_neck_module.training = \
        label_neck_module.training = \
        decoder_module.training = \
        patch_adapter_module.training = \
        label_adapter_module.training = \
        patch_proj_module.training = \
        label_proj_module.training = is_training

    # build model
    model = aio_entry_v2mae_shareneck(backbone_module,
                                      patch_neck_module,
                                      label_neck_module,
                                      decoder_module,
                                      patch_adapter_module,
                                      label_adapter_module,
                                      patch_proj_module,
                                      label_proj_module,
                                      modalities,
                                      config.get('model_entry_kwargs', {}),)
    model.training = is_training
    return model


def module_compare(module_list):
    for i, j in zip(module_list[0].state_dict(), module_list[1].state_dict()):
        if module_list[0].state_dict()[i].size() != module_list[1].state_dict()[i].size():
            print(module_list[0].state_dict()[i].size(),
                  module_list[1].state_dict()[i].size())
            print(i)


def transform_image(input_image, transform, device):
    img = transform(input_image)
    img = img.unsqueeze(0).to(device)
    return img


class HumanHulk:
    def __init__(self,
                 device,
                 pose_active=True,
                 parse_active=True,
                 caption_active=False,
                 margin=10):

        self.device = device
        self.margin = margin
        self.pose_active = pose_active
        self.parse_active = parse_active
        self.caption_active = caption_active
        config = './experiments/release/custom_config.yaml'
        # DETECT
        path_detect = 'checkpoints/ckpt_task4_iter_newest.pth.tar'
        self.det_model = self.load_model(config, path_detect, 0)
        self.det_transform = transforms.Compose([
                        transforms.PILToTensor(),
                        ])
        # POSE
        if pose_active:
            path_pose = 'checkpoints/ckpt_task8_iter_newest.pth.tar'
            self.pose_model = self.load_model(config, path_pose, 1)
            self.pose_transform = transforms.Compose([
                            transforms.Resize((256, 192)),
                            transforms.PILToTensor(),
                            ])
            self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
                               (13, 14), (15, 16)]
        # PARSE
        if parse_active:
            path_parse = 'checkpoints/ckpt_task18_iter_newest.pth.tar'
            self.parse_model = self.load_model(config, path_parse, 2)
            CIHP_palette = np.array([[0, 0, 0],
                                     [128, 0, 0],
                                     [255, 0, 0],
                                     [0, 85, 0],
                                     [170, 0, 51],
                                     [255, 85, 0],
                                     [0, 0, 85],
                                     [0, 119, 221],
                                     [85, 85, 0],
                                     [0, 85, 85],
                                     [85, 51, 0],
                                     [52, 86, 128],
                                     [0, 128, 0],
                                     [0, 0, 255],
                                     [51, 170, 221],
                                     [0, 255, 255],
                                     [85, 255, 170],
                                     [170, 255, 85],
                                     [255, 255, 0],
                                     [255, 170, 0]])
            self.palette = CIHP_palette.flatten().tolist()
            self.parse_transform = transforms.Compose([
                            transforms.Resize(480),
                            transforms.PILToTensor(),
                            ])
        # CAPTION
        if caption_active:
            from transformers import BertTokenizer
            path_caption = 'checkpoints/ckpt_task7_iter_newest.pth.tar'
            self.caption_model = self.load_model(config, path_caption, 3)
            self.tokenizer = BertTokenizer.from_pretrained(
                './experiments/release/bert-base-uncased/', do_lower=True)
            self.caption_transform = transforms.Compose([
                            transforms.Resize(384),
                            transforms.ToTensor(),
                            ])

    def load_model(self, config, checkpoint_path, task_idx):
        C_hulk = Config_Hulk(config, task_idx=task_idx, noginfo=True)
        C_hulk.config['common']['model_entry_kwargs']['test_flag'] = C_hulk.config['common']['model_entry_kwargs']['test_flag'][task_idx]

        config = edict(C_hulk.config['common'])
        model = create_model(config, self.device)

        pose_ckpt = torch.load(checkpoint_path, map_location=self.device)
        pose_ckpt = pose_ckpt['state_dict']
        for key in list(pose_ckpt.keys()):
            pose_ckpt[key[7:]] = pose_ckpt.pop(key)

        model.load_state_dict(pose_ckpt)
        model.to(self.device)
        return model

    def set_image(self, img_path, max_size=1024):
        self.img = Image.open(img_path).convert('RGB')
        self.source_W, self.source_H = self.img.size

        if self.source_W > max_size or self.source_H > max_size:
            if self.source_W > self.source_H:
                self.W = max_size
                self.H = self.source_H*max_size//self.source_W
            elif self.source_W < self.source_H:
                self.H = max_size
                self.W = self.source_W*max_size//self.source_H
            else:
                self.H = max_size
                self.W = max_size
            self.img = self.img.resize((self.W, self.H))
            self.resized = True
        else:
            self.W = self.source_W
            self.H = self.source_H
            self.resized = False

        self.box, self.detected_human = self.get_detection()

        if self.box is not None:
            if self.resized:
                box = self.box*[self.source_W/self.W, self.source_W /
                                self.W, self.source_H/self.H, self.source_H/self.H]
            else:
                box = self.box

            box = (np.int32(box)).tolist()[0]
        else:
            box = None
        return box

    def get_detection(self):
        og_img = self.img.copy()
        img = transform_image(self.img, self.det_transform, self.device)
        mask = torch.zeros(1, self.H, self.W).to(self.device).to(bool)
        sparse_labeling = torch.zeros(1, 3, 2, 867, 1).to(self.device)
        orig_size = torch.tensor([[self.H, self.W]]).to(self.device)
        img = NestedTensor(img, mask=mask)
        input_img = edict(image=img,
                          sparse_labeling=sparse_labeling,
                          orig_size=orig_size)
        output = self.det_model(input_img, 0)
        threshold = 0.5
        det_idx = (output['pred'][0]['scores'] >
                   threshold).nonzero(as_tuple=True)
        det_idx = det_idx[0]

        if det_idx.shape[0] == 0:
            det_idx = [0]
            return None, None
        boxes = output['pred'][0]['boxes'][det_idx].cpu().detach()
        boxes[:, :2] -= self.margin
        boxes[:, 2:4] += self.margin

        boxes[:, :2][boxes[:, :2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > self.W] = self.W
        boxes[:, 3][boxes[:, 3] > self.H] = self.H
        boxes = boxes.numpy()
        boxes = np.int32(boxes)
        cropped_img = og_img.crop(boxes[0])
        return boxes, cropped_img

    def get_pose(self, img_path, radius=3):
        assert self.pose_active, '2D pose model has not been loaded, load it first'
        assert self.detected_human is not None or self.box is not None, 'Human not detected, try another image'
        img = transform_image(self.detected_human, self.pose_transform, self.device)
        cW, cH = self.detected_human.size
        center = [cW/2, cH/2]
        scale = [cW/200, cH/200]
        input_img = edict(image=img,
                          img_metas=[edict(
                              data=edict(
                                  flip_pairs=self.flip_pairs,
                                  center=center,
                                  scale=scale,
                                  image_file=img_path
                              ))])
        output = self.pose_model(input_img, 0)
        keypoints = mmpose_to_coco(output['preds'])
        keypoints += (keypoints != -1)*self.box[0][:2][::-1]

        if self.resized:
            keypoints = (keypoints != -1)*keypoints * \
                [self.source_W/self.W, self.source_H/self.H] + \
                (keypoints == -1)*(-1)
            radius = int(radius*self.source_W/self.W)
        pose = draw_pose_from_cords(
            keypoints, (self.source_H, self.source_W), radius=radius, draw_bones=True)
        pose = Image.fromarray(pose)
        keypoints = keypoints.tolist()
        return keypoints, pose

    def get_parse(self):
        assert self.parse_active, 'CIHP Parse model has not been loaded, load it first'
        assert self.detected_human is not None or self.box is not None, 'Human not detected, try another image'

        cW, cH = self.detected_human.size
        img = transform_image(self.detected_human, self.parse_transform, self.device)
        input_img = edict(image=img,
                          height=torch.tensor([cH]), width=torch.tensor([cW]))
        output = self.parse_model(input_img, 0)
        parse = output['pred'][0]['sem_seg']
        zeros = torch.zeros((1, 20, self.H, self.W))
        zeros[:, :, self.box[0][1]:self.box[0][3],
              self.box[0][0]:self.box[0][2]] = parse

        if self.resized:
            zeros = F.interpolate(
                zeros, (self.source_H, self.source_W), mode='bilinear', align_corners=False
            )
        zeros = zeros.squeeze(0)
        zeros = torch.argmax(zeros, dim=0)
        img = Image.fromarray(np.uint8(zeros))
        img.putpalette(self.palette)
        return img

    def get_caption(self):
        assert self.caption_active, 'Caption model has not been loaded, load it first'
        assert self.detected_human is not None or self.box is not None, 'Human not detected, try another image'
        img = transform_image(self.detected_human, self.caption_transform, self.device)
        max_words = 40
        caption_id = torch.zeros((1, max_words), dtype=torch.int32)
        token_type_id = torch.zeros((1, max_words), dtype=torch.int32)
        caption_pad_mask = torch.zeros((1, max_words), dtype=torch.int32)

        input = edict(image=img, input_id=caption_id,
                      token_type_id=token_type_id, padding_mask=caption_pad_mask)

        output = self.caption_model(input, 0)
        output = self.tokenizer.decode(output['pred'].squeeze(
            0).tolist(), skip_special_tokens=True)
        return output
