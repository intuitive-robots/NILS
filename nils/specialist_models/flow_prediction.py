import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from unimatch.unimatch import UniMatch

from nils.specialist_models.detectors.utils import create_batches
from nils.utils.flow_viz import flow_to_image

device  = 'cuda' if torch.cuda.is_available() else 'cpu'

from pathlib import Path

class FlowPredictor:

    def __init__(self,
                 unimatch_checkpoint_path= os.path.join(os.environ["NILS_DIR"], "dependencies","unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")):

        task = "flow"
        self.flow_model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task=task)
        checkpoint_flow = torch.load(unimatch_checkpoint_path)
        self.flow_model.load_state_dict(checkpoint_flow['model'], strict=True)
        self.flow_model = self.flow_model.cuda().eval()


        self.padding_factor = 32
        self.attn_type = 'swin' if task == 'flow' else 'self_swin2d_cross_swin1d'
        self.attn_splits_list = [2, 8]
        self.corr_radius_list = [-1, 4]
        self.prop_radius_list = [-1, 1]
        self.num_reg_refine = 6 if task == 'flow' else 3
        self.max_inference_size = [384, 768] if task == 'flow' else [384, 768]
        #self.inference_size = (320,576)
        self.inference_size = None
        self.optical_flow_interval = 4
        self.task = task


    def to_cpu(self):
        self.flow_model = self.flow_model.to('cpu')
        torch.cuda.empty_cache()


    def calc_optical_flow(self, frames, optical_flow_interval,bsz = 1):

        self.flow_model.to(device)

        frames = frames[::optical_flow_interval]

        if frames.ndim == 3:
            frames = frames[..., None]

        if len(frames) % optical_flow_interval != 0 and len(frames) > 2:
            frames = np.concatenate([frames, frames[-1:]], axis=0)

        frames_batched = []
        for frame in range(len(frames) - 1):
            frames_cat = np.stack([frames[frame], frames[frame + 1]], axis=0)
            frames_batched.append(frames_cat)
        frames_batched = np.stack(frames_batched, axis=0)

        # orig_size = frames[0].shape[-2:]

        #bsz = 1
        frames_batched = create_batches(bsz, frames_batched)

        res_fwd = []
        res_bwd = []
        
        flows_fwd = []
        flows_bwd = []
        
        for batch in tqdm(frames_batched):
            start_frames = batch[:, 0]
            next_frames = batch[:, 1]
            image1 = np.array(start_frames).astype(np.float32)
            image2 = np.array(next_frames).astype(np.float32)
            image1 = torch.from_numpy(image1).permute(0, 3, 1, 2).float()
            image2 = torch.from_numpy(image2).permute(0, 3, 1, 2).float()
            
            #image1 = F.interpolate(image1, size=self.inference_size, mode='bilinear')
            #image2 = F.interpolate(image2, size=self.inference_size, mode='bilinear')

            transpose_img = False
            if image1.size(-2) > image1.size(-1):
                image1 = torch.transpose(image1, -2, -1)
                image2 = torch.transpose(image2, -2, -1)
                transpose_img = True
            
            if self.inference_size is None:
                nearest_size = [int(np.ceil(image1.size(-2) / self.padding_factor)) * self.padding_factor,
                                int(np.ceil(image1.size(-1) / self.padding_factor)) * self.padding_factor]
    
                inference_size = [min(self.max_inference_size[0], nearest_size[0]),
                                  min(self.max_inference_size[1], nearest_size[1])]
            else:
                inference_size = self.inference_size

            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]

            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='nearest',
                                       ).cuda()
                image2 = F.interpolate(image2, size=inference_size, mode='nearest',
                                       ).cuda()

            with torch.inference_mode():
                #with torch.autocast(device_type="cuda", dtype=torch.float16):
                results_dict = self.flow_model(image1, image2,
                                               attn_type=self.attn_type,
                                               attn_splits_list=self.attn_splits_list,
                                               corr_radius_list=self.corr_radius_list,
                                               prop_radius_list=self.prop_radius_list,
                                               num_reg_refine=self.num_reg_refine,
                                               pred_bidir_flow=True,
                                               task="flow",
                                               )

            flow_pr = results_dict['flow_preds'][-1]  # [1, 2, H, W] or [1, H, W]

            # rself.esize back
            if self.task == 'flow':
                if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                    flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                            align_corners=True)
                    flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
                    flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

                if transpose_img:
                    flow_pr = torch.transpose(flow_pr, -2, -1)
                flow = flow_pr.permute(0, 2, 3, 1).cpu().numpy()
                output = flow_to_image(flow[0])
            output = np.abs(flow).sum(axis=-1)
            flow_masks = output > output.mean(axis=(-1, -2))[..., None, None] * 1.4
            no_movement_mask = output.max(axis=(-1, -2)) > 2
            flow_masks = np.logical_and(flow_masks, no_movement_mask[..., None, None])



            flow_masks_fwd = flow_masks[:len(image1)]
            flow_masks_bwd = flow_masks[len(image1):]
            flows_fwd.append(flow[:len(image1)])
            flows_bwd.append(flow[len(image1):])

            res_fwd.append(flow_masks_fwd)
            res_bwd.append(flow_masks_bwd)

        flow_masks_fwd = np.concatenate(res_fwd, axis=0)
        flow_masks_bwd = np.concatenate(res_bwd, axis=0)
        
        flows_fwd = np.concatenate(flows_fwd, axis=0)
        flows_bwd = np.concatenate(flows_bwd, axis=0)

        return flow_masks_fwd, flow_masks_bwd, flows_fwd, flows_bwd
