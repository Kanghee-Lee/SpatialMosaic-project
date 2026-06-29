import argparse
import json
import logging
import math
import os
import signal
from collections import defaultdict
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("lmms-eval")
import sys
import os

VILA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "VILA"))
print(f"DEBUG: Current working directory: {os.getcwd()}")
sys.path.insert(0, VILA_ROOT)
print(f"DEBUG: sys.path after inserting 'VILA/': {sys.path}")
print(f"DEBUG: Absolute path for 'VILA/': {VILA_ROOT}")
print(f"DEBUG: Does VILA/llava exist at this path? {os.path.exists(os.path.join(VILA_ROOT, 'llava'))}")

try:
    print("DEBUG: Attempting to import from llava...")
    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
    )
    try:
        from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    except ImportError:
        DEFAULT_IM_END_TOKEN = None
        DEFAULT_IM_START_TOKEN = None
        IMAGE_TOKEN_INDEX = None
    print("DEBUG: Imported llava.constants")
    from llava import conversation as conversation_lib
    from llava.conversation import SeparatorStyle, conv_templates
    print("DEBUG: Imported llava.conversation")
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    print("DEBUG: Imported llava.mm_utils")
    from llava.model.builder import load_pretrained_model
    print("DEBUG: Imported llava.model.builder")
except ImportError as e:
    print(f"DEBUG: ImportError occurred: {e}")
    eval_logger.debug(f"VILA is not installed. Please install VILA to use this model. Error: {e}")


def prepare_config_for_eval(config, kwargs):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")

    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    # NOTE: seems siglip works well with device_map = "auto"
    # vision_tower_name = parse_model_name_or_path(config, "vision_tower")
    # if "siglip" in vision_tower_name.lower():
    #     kwargs["device_map"] = "cuda"

import llava.model.builder
llava.model.builder.prepare_config_for_eval = prepare_config_for_eval


@register_model("vila")
class VILA(lmms):
    """
    VILA Model
    """

    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/VILA1.5-40b",
        model_base: Optional[str] = None,
        max_frames_num: Optional[int] = 100,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        torch_dtype: Optional[str] = "bfloat16",
        device_map="cuda:0",
        conv_template="hermes-2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        video_decode_backend="decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = max_frames_num
        # self._config = AutoConfig.from_pretrained(self.pretrained)

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, self.model_name, model_base, device_map=self.device_map, attn_implementation=attn_implementation, torch_dtype=torch_dtype)

        self.model.image_processor = self._image_processor
        
        
        self._config = self._model.config

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.video_decode_backend = video_decode_backend
        self.model.eval()
        # self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        conversation_lib.default_conversation = conv_templates[self.conv_template].copy()
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def _tokenizer_image_token(self, prompt):
        if IMAGE_TOKEN_INDEX is None:
            return tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        return tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return [Image.fromarray(img) for img in spare_frames]
        except Exception as e:
            eval_logger.error(f"Failed to load video {video_path} with error: {e}")
            return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            videos = []
            for visual in visuals:
                video = self.load_video(visual, self.max_frames_num)
                video = self._image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
                videos.append(video)

            qs = contexts
            if getattr(self.model.config, "mm_use_im_start_end", False) and DEFAULT_IM_START_TOKEN is not None and DEFAULT_IM_END_TOKEN is not None:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            contxt_id = self._tokenizer_image_token(prompt).unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            input_ids = self._tokenizer_image_token(prompt).unsqueeze(0).cuda()
            attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=videos, modalities="video")

            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _visual_to_pil(self, visual):
        if isinstance(visual, Image.Image):
            return visual.convert("RGB")
        if torch.is_tensor(visual):
            visual = visual.detach().cpu().numpy()
        if isinstance(visual, np.ndarray):
            if visual.ndim == 3 and visual.shape[0] in (1, 3) and visual.shape[-1] not in (1, 3, 4):
                visual = np.transpose(visual, (1, 2, 0))
            return Image.fromarray(visual.astype(np.uint8)).convert("RGB")
        raise NotImplementedError(f"Unsupported visual type: {type(visual)}")

    def _frame_stack_to_images(self, visual):
        if not isinstance(visual, np.ndarray):
            return None
        if visual.ndim == 3:
            visual = np.expand_dims(visual, axis=0)
        if visual.ndim != 4:
            raise ValueError(f"Expected frame stack with 3 or 4 dims, got shape {visual.shape}")
        if self.max_frames_num and visual.shape[0] > self.max_frames_num:
            indices = np.linspace(0, visual.shape[0] - 1, self.max_frames_num, dtype=int)
            visual = visual[indices]
        return [self._visual_to_pil(frame) for frame in visual]

    def _visual_to_prompt_parts(self, visual):
        if visual is None:
            return []

        frame_stack_images = self._frame_stack_to_images(visual)
        if frame_stack_images is not None:
            return frame_stack_images

        visuals = self.flatten([visual])
        if not visuals:
            return []

        if isinstance(visuals[0], str):
            num_video_frames = self.model.config.num_video_frames
            if self.max_frames_num and self.max_frames_num < num_video_frames:
                num_video_frames = self.max_frames_num
            if self.max_frames_num == 0:
                return [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

            images = []
            for visual in visuals:
                if self.video_decode_backend == "decord":
                    images.extend(self.load_video(visual, num_video_frames))
                elif self.video_decode_backend == "pyav":
                    images.extend(read_video_pyav(visual, num_frm=num_video_frames))
                else:
                    raise ValueError(f"Unsupported video_decode_backend: {self.video_decode_backend}")
            return images

        if isinstance(visuals[0], Image.Image):
            return [self._visual_to_pil(visual) for visual in visuals]

        raise NotImplementedError(f"Unsupported visual type: {type(visuals[0])}")

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visual = doc_to_visual(self.task_dict[task][split][doc_id]) if doc_to_visual is not None else None
            prompt_parts = self._visual_to_prompt_parts(visual) + [contexts]

            generation_kwargs = dict(gen_kwargs)
            generation_kwargs.pop("until", None)
            if "max_new_tokens" not in generation_kwargs:
                generation_kwargs["max_new_tokens"] = 1024
            if "temperature" not in generation_kwargs:
                generation_kwargs["temperature"] = 0.2
            if "top_p" not in generation_kwargs:
                generation_kwargs["top_p"] = None
            if "num_beams" not in generation_kwargs:
                generation_kwargs["num_beams"] = 1
            temperature = generation_kwargs.get("temperature")
            generation_kwargs.setdefault("do_sample", bool(temperature and temperature > 0))
            generation_kwargs.setdefault("use_cache", self.use_cache)
            if not generation_kwargs["do_sample"]:
                generation_kwargs.pop("temperature", None)
                generation_kwargs.pop("top_p", None)

            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            with torch.inference_mode():
                outputs = self.model.generate_content(prompt_parts, generation_config=generation_config)

            res.append(outputs)
            pbar.update(1)
        return res
