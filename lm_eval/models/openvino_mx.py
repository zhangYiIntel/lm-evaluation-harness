import logging
from importlib.util import find_spec

import numpy as np
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger(__name__)


@register_model("openvino-mx")
class OpenVINOMX(LM):
    """OpenVINO VLM backend using VLMPipeline.

    This backend is intended for multimodal generation tasks where
    `Instance.args` is `(context, gen_kwargs, aux_arguments)` and
    `aux_arguments["visual"]` contains image objects.
    """

    MULTIMODAL = True

    def __init__(
        self,
        pretrained: str,
        device: str = "CPU",
        max_new_tokens: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.device = device
        self.default_max_new_tokens = int(max_new_tokens)

        if not find_spec("openvino"):
            raise ModuleNotFoundError(
                "package `openvino` is not installed. Please install it first."
            )

        if not find_spec("pipeline"):
            raise ModuleNotFoundError(
                "Could not import `pipeline`. Ensure your VLMPipeline module is importable."
            )

        import openvino as ov
        from pipeline import GenerationConfig, VLMPipeline

        self.ov = ov
        self.GenerationConfig = GenerationConfig
        self.pipe = VLMPipeline(self.pretrained, self.device)

    def _images_to_tensors(self, aux_arguments: dict | None):
        if not aux_arguments:
            return None

        visuals = aux_arguments.get("visual")
        if not visuals:
            return None

        tensors = []
        for image in visuals:
            if isinstance(image, self.ov.Tensor):
                tensors.append(image)
                continue

            # Most multimodal tasks pass PIL Images. Convert to HWC uint8 tensor.
            image_np = np.array(image)
            tensors.append(self.ov.Tensor(image_np))

        return tensors

    @staticmethod
    def _apply_stop(text: str, until: list[str] | None) -> str:
        if not until:
            return text
        out = text
        for term in until:
            if term:
                out = out.split(term)[0]
        return out

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        if not requests:
            return []

        results = []
        iterator = tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running OpenVINO VLM generate_until requests",
        )

        for args in iterator:
            if len(args) >= 3:
                context, gen_kwargs, aux_arguments = args
            elif len(args) == 2:
                context, gen_kwargs = args
                aux_arguments = None
            else:
                context = args[0]
                gen_kwargs = {}
                aux_arguments = None

            if not isinstance(gen_kwargs, dict):
                raise ValueError(
                    f"Expected generation kwargs as dict, got {type(gen_kwargs)}"
                )

            cfg = self.GenerationConfig()
            cfg.max_new_tokens = int(
                gen_kwargs.get("max_gen_toks", self.default_max_new_tokens)
            )

            until = gen_kwargs.get("until")
            if until is not None and not isinstance(until, list):
                until = [until]

            image_tensors = self._images_to_tensors(aux_arguments)
            if image_tensors is None:
                raise ValueError(
                    "openvino-mx requires image input in aux_arguments['visual']."
                )

            output = self.pipe.generate(context, image_tensors, cfg)
            text = output.texts[0] if output and output.texts else ""
            results.append(self._apply_stop(text, until))

        return results

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "openvino-mx currently supports generate_until only."
        )

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        raise NotImplementedError(
            "openvino-mx currently supports generate_until only."
        )