import re

import logging
import torch
import transformers
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model

transformers.set_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


class LLAVA():

    def __init__(self):

        hf_model_name = "liuhaotian/llava-v1.5-13b"
        model_name = get_model_name_from_path(hf_model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(hf_model_name, None,
                                                                                                   model_name, False,
                                                                                                   True, device=device)

        self.temperature = 0.2
        self.top_p = 0.7
        self.num_beams = 1

    def ids_to_tokens(self, ids):
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def inference(self, prompt, image, return_top_k=None):
        qs = prompt
        cur_prompt = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        self.model.eval()
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = {}
        prompt = prompt

        args = {}
        args["image_aspect_ratio"] = "pad"
        if image.ndim == 3:
            image = image[None, ...]
        image = list(image)
        image_tensor = process_images(image, self.image_processor, args)
        if len(image) > 1:
            image_tensor = image_tensor.unsqueeze(0)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inputs["input_ids"] = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                    return_tensors='pt').unsqueeze(0).cuda()
        targets = {}
        # target_ids = self.tokenizer("H", add_special_tokens=False, return_tensors="pt").input_ids.to(
        #    inputs["input_ids"].device)

        # inputs["input_ids"] = torch.cat([inputs["input_ids"], target_ids], -1)

        # inputs["input_ids"] = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        input_ids = inputs["input_ids"]
        # inputs["attention_mask"] = self.tokenizer(prompt,return_tensors='pt')["attention_mask"]
        # labels = inputs["input_ids"].clone()
        # labels[:, :-target_ids.shape[0]] = -100
        # inputs["labels"] = labels

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs["input_ids"])

        # image_tensor = einops.repeat(image_tensor, 'b c h w -> b t c h w', t = 2)

        with torch.inference_mode():

            # outputs = self.model(input_ids = inputs["input_ids"],labels=inputs["labels"],images = image_tensor.unsqueeze(0).half().cuda())
            ##Method one:
            # outputs.logits[:, -2, :].softmax(-1).max()
            # (-1*outputs.loss).exp()
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                # num_return_sequences = 2,

                # temperature=0.2,
                max_new_tokens=512,
                use_cache=True,
                # stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True
            )
            # Method two (uses temperature and top p sampling, not desired for our case):

            output_ids = outputs["sequences"]
            gen_ids = output_ids[:, input_ids.shape[1]:]

            # check that last token is </s>

            probs = []

        answer_re = re.compile("^[A-Z]\s*:.*")
        input_token_len = input_ids.shape[1]
        scores = torch.stack(outputs["scores"], dim=1)
        outputs_text = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        for i, el in enumerate(outputs["sequences"]):
            gen_ids = el[input_ids.shape[1]:]
            gen_ids = gen_ids[gen_ids != 2]
            if len(gen_ids) > 1:
                logging.error("Model output longer sequence, skipping datapoint ")

                if answer_re.match(outputs_text[i]):
                    probs.append(scores[i, 0, :].softmax(-1).cpu())

                # probs.append(False)
            else:
                probs.append(scores[i, 0, :].softmax(-1).cpu())

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs_cleaned = []
        for output in outputs_text:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            outputs_cleaned.append(output)
        return outputs_cleaned, probs
