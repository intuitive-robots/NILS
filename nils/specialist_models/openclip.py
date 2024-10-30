import os

import numpy as np
import open_clip
import torch
from PIL import Image

from nils.specialist_models.clip_utils import (
    get_text_classifier,
    get_text_classifier_ov_sam,
)
from nils.specialist_models.detectors.utils import create_batches

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prompt_templates = [
    "This is a picture of a [state] [obj].",
    "An image of a [state] [obj].",
    "A picture of a [state] [obj].",
    "A photo of a [obj] that is [state].",
    "A [state] [obj].",
    "[state]"
    #"A [state] compartment",
    #"A [state] object"
    #"A robot in front of a [state] [obj].",
    #"A picture of a [obj], the door is [state] and the inside is visible",
    #"An image of a partially occluded [state] [obj].",
]

VILD_PROMPT = [
    "an image of a {}.",
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "an image of a {} in the scene",
    "a picture of a small {}.",
    "a phot of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

class OpenClip:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k',tokenizer = None, bsz = 16):
        #self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        
        self.model_name = model_name
        
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)

        self.model = self.model.to(device)

        
        
        #transforms = preprocess.transforms
        
        #del transforms[2]
        
        #self.preprocess = Compose(transforms)
        
        
        
        self.tokenizer = open_clip.get_tokenizer(model_name if tokenizer is None else tokenizer)
        
        self.bsz = bsz
        
        self.text_classifier = None
        self.classes = None
        self.logit_scale = 100
        
        self.orig_name = model_name
        self.orig_tokenizer = tokenizer
        
        
    def set_model(self,model_name, tokenizer_name):
        
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_name)
        self.tokenizer = open_clip.get_tokenizer(tokenizer_name)
        
    def set_to_orig_model(self):
        self.model, self.preprocess = open_clip.create_model_from_pretrained(self.orig_name)
        self.tokenizer = open_clip.get_tokenizer(self.orig_tokenizer)
        
        
    
    def get_video_similarity(self,frames,prompts,reduction = "none"):
        
        self.model = self.model.to(device)

        batch_size = self.bsz

        #image_batches = create_batches(batch_size, frames)

        
        text = self.tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = self.model.encode_text(text)

      
        

        text_probs = []

        text_features = text_features.to(device)

        logit_bias = self.model.logit_bias

        if logit_bias is None:
            logit_bias = 0

        batch = frames
        batch = torch.stack([self.preprocess(Image.fromarray(image)) for image in batch])
        batch = batch.to(device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            #average across time dimension
            image_features = torch.mean(image_features,dim = 0)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.to(torch.float32)
            text_features /= text_features.norm(dim=-1, keepdim=True).to(torch.float32)
            text_features = text_features.to(torch.float32)

            # Debug
            if text_features.ndim == 3:
                # take max over class synonyms
                cos_sim = torch.einsum('bc,nkc->bnk', image_features, text_features)
                cos_sim = torch.max(cos_sim, dim=-1).values
                cos_sim = cos_sim * self.model.logit_scale.exp() + logit_bias

            else:
                cos_sim = (image_features @ text_features.T) * self.model.logit_scale.exp() + logit_bias

            # res  = []
            # for i in  range(len(prompt_templates)):
            #     to_compare = torch.stack([cos_sim[:,i],cos_sim[:,len(prompt_templates)+i]],dim = 1).softmax(dim=-1)
            #     res.append(to_compare)

            # text_probs_batch = torch.cat(res, dim=1)

            if reduction == "sig":
                text_probs_batch = cos_sim.sigmoid()
            elif reduction == "softmax":
                text_probs_batch = cos_sim.softmax(dim=1)
            elif reduction == "none":
                text_probs_batch = cos_sim
            else:
                raise ValueError("Unknown reduction method")
            text_probs_batch = text_probs_batch.cpu()

            text_probs.append(text_probs_batch)

            del image_features, cos_sim
            torch.cuda.empty_cache()

        text_probs = text_probs_batch.cpu().numpy()

        with torch.no_grad():
            del text_features
            torch.cuda.empty_cache()

        return text_probs
    

    def get_preds(self,cropped,prompts = None,precomputed_embeddings = None, reduction = "sig",bsz = 16):
        
        
        self.model = self.model.to(device)
        
        batch_size = bsz

        image_batches = create_batches(batch_size, cropped)
        
        if precomputed_embeddings is None:
            text = self.tokenizer(prompts).to(device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
            
            if len(prompts) % len(prompt_templates) == 0:
                text_features_split_class = torch.stack(torch.split(text_features, len(prompt_templates)))
                text_features_split_class /= text_features_split_class.norm(dim=-1, keepdim=True)
                text_features = torch.mean(text_features_split_class, dim=1)
        else:
            text_features = precomputed_embeddings

        text_probs = []
        
        text_features = text_features.to(device)
        
        logit_bias = self.model.logit_bias
        
        if logit_bias is None:
            logit_bias = 0

        for batch in image_batches:
            batch = torch.stack([self.preprocess(Image.fromarray(image)) for image in batch])
            batch = batch.to(device)
            with torch.no_grad():
                image_features = self.model.encode_image(batch)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.to(torch.float32)
                text_features /= text_features.norm(dim=-1, keepdim=True).to(torch.float32)
                text_features = text_features.to(torch.float32)
            
                # Debug
                if text_features.ndim == 3:
                    #take max over class synonyms
                    cos_sim = torch.einsum('bc,nkc->bnk', image_features, text_features)
                    cos_sim = torch.max(cos_sim, dim=-1).values
                    cos_sim =   cos_sim * self.model.logit_scale.exp() + logit_bias
                   
                else:
                    cos_sim = (image_features @ text_features.T) * self.model.logit_scale.exp() + logit_bias
                
                

                # res  = []
                # for i in  range(len(prompt_templates)):
                #     to_compare = torch.stack([cos_sim[:,i],cos_sim[:,len(prompt_templates)+i]],dim = 1).softmax(dim=-1)
                #     res.append(to_compare)

                #text_probs_batch = torch.cat(res, dim=1)
                
                if reduction == "sig":
                    text_probs_batch = cos_sim.sigmoid()
                elif reduction == "softmax":
                    text_probs_batch = cos_sim.softmax(dim = 1)
                elif reduction == "none":
                    text_probs_batch = cos_sim
                else:
                    raise ValueError("Unknown reduction method")
                text_probs_batch = text_probs_batch.cpu()

                text_probs.append(text_probs_batch)

                del image_features,cos_sim
                torch.cuda.empty_cache()

        text_probs = torch.cat(text_probs, dim=0).cpu().numpy()
        
        with torch.no_grad():
            del text_features
            torch.cuda.empty_cache()
        

        return text_probs
    
    
    def get_text_embeddings(self,texts):
        
        self.model = self.model.to(device)
            
        with torch.no_grad():
            text = self.tokenizer(texts).to(device)
            text_features = self.model.encode_text(text)
            text_features = text_features.cpu()
        return text_features
    
    def predict(self, cropped,states,obj, reduction = "sig"):
        
        #if self.model.attn_mask.device != device:

        

        
        
        prompts = self.create_prompts(states,obj)
        
        text_probs = self.get_preds(cropped,prompts,reduction = reduction)
        
        del prompts
        torch.cuda.empty_cache()

        return text_probs
    
    
    def predict_objects(self,images,bsz = 16):
        
        
        self.model = self.model.to(device)
            
        if self.text_classifier is not None:
            text_classifier = self.text_classifier
        else:
            text_classifier = None
            
        probs = self.get_preds(images, precomputed_embeddings=text_classifier,bsz = bsz)

        self.model.to('cpu')

        
        return probs
        
        
        
        
    def create_prompts(self,states,obj):

        state_prompts = []
        for state in states:
            
            if "siglip" in self.model_name.lower():
                
                state_prompts.append("[state] [obj]".replace("[state]",state).replace("[obj]",obj))
                
            else:
                for prompt in prompt_templates:
                    state_prompts.append(prompt.replace("[state]",state).replace("[obj]",obj))
        
      
                
        
        return state_prompts
    
    
    
    
    def compute_text_embeddings(self,classes = None,file = None,save_dir = None):
        
        if file is None and classes is None:
            raise ValueError("Either file or classes must be provided")
        if file is not None:
            with open(file) as f:
                lvis_classes = f.read().splitlines()
                lvis_classes = [x[x.find(':') + 1:] for x in lvis_classes]
                classes = lvis_classes
        else:
            classes = classes

        # gpt_lst = gpt_lst
        # 
        
        #lvis_classes = lvis_classes + gpt_lst
        #lvis_classes = np.unique(lvis_classes)
        
        #all_classes = lvis_classes + gpt_list
        all_classes = classes 
        
        self.classes = all_classes

        
        
        if save_dir is not None and os.path.exists(save_dir):
            text_classifier = torch.load(save_dir)
            self.text_classifier = text_classifier
            return text_classifier
        text_classifier = get_text_classifier_ov_sam(self,all_classes,save_dir)
        self.text_classifier = text_classifier
        
        #with open(save_dir,"wb") as f:
        #    torch.save(text_classifier,f)
        
        return text_classifier


def get_occluded_frames(cropped_images, cropped_robot_masks, threshold=0.1):
    occluded_frames = []
    for img, robot_mask in zip(cropped_images, cropped_robot_masks):
        img_n_pixels = img.shape[0] * img.shape[1]
        masked_img = img * robot_mask[..., None]
        masked_img_vis = img * (~robot_mask[..., None].astype(bool))
        overlap = (masked_img > 0).sum()
        if overlap > threshold * img_n_pixels:
            occluded_frames.append(True)
        else:
            occluded_frames.append(False)
    return np.array(occluded_frames)
