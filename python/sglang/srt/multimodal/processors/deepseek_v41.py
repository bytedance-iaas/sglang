"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging
from functools import partial

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    GPU_PLAN_KEY,
    image_token_types,
    patchify_image,
    patchify_image_rust,
    prepare_image_gpu,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm
from sglang.srt.rust_extensions import load_rust_extension

logger = logging.getLogger(__name__)


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False
    detach_epd_embeddings_from_pool = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.preprocess_backend = "cpu"
        self.cpu_patchify = patchify_image
        backend = get_mm().image_processor_backend
        if backend == "pil" or get_mm().disable_fast_image_processor:
            self.preprocess_backend = "cpu"
        elif envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get():
            self.preprocess_backend = "gpu"
        elif backend == "auto":
            # Resolved before super().__init__(), which fingerprints the backend;
            # a load failure falls back to PIL, later image errors do not.
            try:
                extension = load_rust_extension(
                    "sglang.srt.rust_extensions._multimodal"
                )
                resize_patchify = extension.dsv41.resize_patchify
            except (ImportError, OSError, RuntimeError, AttributeError) as error:
                logger.warning(
                    "V4.1 Rust image processor unavailable; using PIL: %s", error
                )
            else:
                self.preprocess_backend = "rust"
                self.cpu_patchify = partial(
                    patchify_image_rust, resize_patchify=resize_patchify
                )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.IM_TOKEN_ID = self.image_token_id
        self.IM_START_TOKEN_ID = None
        self.IM_END_TOKEN_ID = None
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    def get_mm_data(self, prompt, embeddings, **kwargs):
        if not isinstance(prompt, list):
            prompt = self._tokenizer.encode(prompt)
        grids = kwargs.get("img_grid_thw")
        image_embeddings = embeddings.get(Modality.IMAGE)
        if grids is None or image_embeddings is None:
            raise ValueError("DeepSeek-V4.1 EPD requires image grids and embeddings")

        input_ids = []
        items = []
        image_index = 0
        embedding_offset = 0
        ratio = self.hf_config.vision_downsample_ratio
        for token in prompt:
            if token != self.image_token_id:
                input_ids.append(token)
                continue
            if image_index >= len(grids):
                raise ValueError("More image placeholders than encoder grids")
            n_vit_h, n_vit_w = (int(v) for v in grids[image_index].flatten()[-2:])
            token_count = len(
                image_token_types(
                    (n_vit_h + ratio - 1) // ratio,
                    (n_vit_w + ratio - 1) // ratio,
                )
            )
            start = len(input_ids)
            input_ids.extend([self.image_token_id] * token_count)
            embedding_slice = image_embeddings[
                embedding_offset : embedding_offset + token_count
            ].clone()
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, start + token_count - 1)],
                    precomputed_embeddings=embedding_slice,
                    model_specific_data={
                        "n_vit_h": n_vit_h,
                        "n_vit_w": n_vit_w,
                    },
                )
            )
            embedding_offset += token_count
            image_index += 1

        if image_index != len(grids):
            raise ValueError("Fewer image placeholders than encoder grids")
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=items,
            im_token_id=self.image_token_id,
        )

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )
        if ids.count(self.image_token_id) != len(base.images):
            raise ValueError("Image placeholders and images must match")
        images = iter(base.images)
        tokens, items = [], []
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            image = next(images)
            plan = None
            if self.preprocess_backend == "gpu":
                patches, plan, lh, lw = await asyncio.to_thread(
                    prepare_image_gpu, image, self.hf_config
                )
                h = plan["height"] // plan["patch_size"]
                w = plan["width"] // plan["patch_size"]
            else:
                patches, h, w, lh, lw = await asyncio.to_thread(
                    self.cpu_patchify, image, self.hf_config
                )
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            metadata = {"n_vit_h": h, "n_vit_w": w}
            if plan is not None:
                metadata[GPU_PLAN_KEY] = plan
            count = len(image_token_types(lh, lw))
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    offsets=[(start, start + count - 1)],
                    model_specific_data=metadata,
                )
            )
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )
