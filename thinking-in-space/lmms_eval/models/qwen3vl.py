import logging
from typing import List, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

eval_logger = logging.getLogger("eval_logger")

DEFAULT_GEN_KWARGS = dict(
    max_new_tokens=64,
    do_sample=False,
)

TORCH_DTYPE_BY_NUMPY_DTYPE = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
}


@register_model("qwen3vl")
class Qwen3VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        modality: str = "image",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: str = "1",
        max_frames_num: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.path = pretrained
        self.accelerator = Accelerator()
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes

        if torch.cuda.is_available():
            if self._world_size > 1:
                self._device = torch.device(f"cuda:{self.accelerator.local_process_index}")
                if device_map in ("auto", "", None):
                    device_map = f"cuda:{self.accelerator.local_process_index}"
            else:
                self._device = torch.device(device)
        else:
            self._device = torch.device("cpu")
            device_map = None

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.path,
            dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(self.path, use_fast=False)

        batch_size = int(batch_size)
        assert batch_size == 1, f"Batch size should be 1 for Qwen3VL, but got {batch_size}."
        self.batch_size_per_gpu = batch_size

        self.modality = modality
        self.max_frames_num = int(max_frames_num) if max_frames_num is not None else None
        self._config = self._model.config

    @property
    def config(self):
        return self._config

    @property
    def model(self):
        return self._model

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

    def _is_empty_visual(self, visual) -> bool:
        if visual is None:
            return True
        if isinstance(visual, (list, tuple)):
            return len(visual) == 0
        if isinstance(visual, np.ndarray):
            return visual.size == 0
        return False

    def _normalize_visuals(self, visual):
        if self._is_empty_visual(visual):
            return []
        if isinstance(visual, np.ndarray):
            if visual.ndim == 3:
                return [visual]
            if visual.ndim == 4:
                return [frame for frame in visual]
            raise ValueError(f"Unsupported ndarray visual shape: {visual.shape}")
        if isinstance(visual, (list, tuple)):
            return list(visual)
        return [visual]

    def _to_pil_image(self, visual):
        if isinstance(visual, Image.Image):
            return visual.convert("RGB")
        if isinstance(visual, str):
            return Image.open(visual).convert("RGB")
        if isinstance(visual, np.ndarray):
            return Image.fromarray(visual.astype(np.uint8)).convert("RGB")
        raise NotImplementedError(f"Unsupported visual type: {type(visual)}")

    def _model_input_device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _numpy_to_tensor(self, array):
        if isinstance(array, torch.Tensor):
            return array
        array = np.ascontiguousarray(array)
        dtype = TORCH_DTYPE_BY_NUMPY_DTYPE.get(array.dtype)
        if dtype is None:
            raise TypeError(f"Unsupported numpy dtype for Qwen3VL input: {array.dtype}")
        tensor = torch.frombuffer(bytearray(array.tobytes()), dtype=dtype)
        return tensor.view(*array.shape).clone()

    def _clean_gen_kwargs(self, gen_kwargs):
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.pop("until", None)
        for key, value in DEFAULT_GEN_KWARGS.items():
            gen_kwargs.setdefault(key, value)
        if gen_kwargs.get("temperature", None) == 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
        return gen_kwargs

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            gen_kwargs = self._clean_gen_kwargs(gen_kwargs)
            visuals = self._normalize_visuals(doc_to_visual(self.task_dict[task][split][doc_id]))

            if self.modality != "image":
                raise NotImplementedError("Qwen3VL currently supports image modality in this evaluator.")

            if self.max_frames_num:
                visuals = visuals[: self.max_frames_num]

            content = [{"type": "image", "image": self._to_pil_image(visual)} for visual in visuals]
            content.append({"type": "text", "text": contexts})
            messages = [{"role": "user", "content": content}]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="np",
            )
            device = self._model_input_device()
            inputs = {key: self._numpy_to_tensor(value).to(device) for key, value in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            res.append(response)
            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "Not implemented yet."
