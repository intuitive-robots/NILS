
import os
from pathlib import Path

import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model




from nils.specialist_models.detectors.utils import create_batches

home = str(Path.home())



device = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "nyu"

user_home = os.path.expanduser("~")

class DepthAnything:
    def __init__(self, model_path = "LiheYoung/depth-anything-large-hf"):

        # model_name = "zoedepth"
        # path = f"local::{home}/Depth-Anything/depth_anything_metric_depth_indoor.pt"
        # #self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        # #self.model = AutoModelForDepthEstimation.from_pretrained(model_path).to(device)
        #
        # config = get_config(model_name, "eval")
        # config.pretrained_resource = path
        # self.model = build_model(config).to(device)
        # self.model.eval()


        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.model.load_state_dict(
            torch.load(os.path.join(user_home,"Depth-Anything-V2",f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth')))

        self.model = self.model.to(device)

        self.model.eval()



        self.bsz = 16

    def to_cpu(self):
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()


    def predict_depth(self, images):

        self.model = self.model.to(device)

        image_batches = create_batches(self.bsz,images)

        depths = []
        for batch in image_batches:

            #original_width, original_height = batch[0].shape[1], batch[0].shape[0]
            #image_tensor = torch.cat([transforms.ToTensor()(np.array(img)).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu') for img in batch])

            image_np = np.array(batch)

            images_transformed = []
            for img in image_np:
                img, (h,w) = self.model.image2tensor(img)
                images_transformed.append(img)
            image_tensor = torch.cat(images_transformed)


            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred = self.model.forward(image_tensor)

                    if isinstance(pred, dict):
                        pred = pred.get('metric_depth', pred.get('out'))
                    elif isinstance(pred, (list, tuple)):
                        pred = pred[-1]




                    pred = torch.nn.functional.interpolate(
                        pred[:, None],
                        size=batch[0].shape[:2],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)

                    pred = pred.detach().cpu()


                    depths.append(pred)
                    del pred
                    torch.cuda.empty_cache()


        return torch.cat(depths, dim=0).numpy().astype(np.float32)



    # def predict_depth(self, images):
    #     image_batches = create_batches(self.bsz,images)
    #
    #     depths = []
    #     for batch in image_batches:
    #         inputs = self.image_processor(batch, return_tensors="pt")
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
    #         with torch.no_grad():
    #             with torch.cuda.amp.autocast():
    #                 outputs = self.model(**inputs)
    #                 predicted_depth = outputs.predicted_depth.cpu().to(torch.float32)
    #                 prediction = torch.nn.functional.interpolate(
    #                     predicted_depth.unsqueeze(1),
    #                     size=batch[0].shape[:2],
    #                     mode="bicubic",
    #                     align_corners=False,
    #                 ).squeeze(1)
    #
    #                 depths.append(prediction)
    #                 del outputs, predicted_depth, prediction
    #                 torch.cuda.empty_cache()
    #
    #     return torch.cat(depths, dim=0).numpy()
