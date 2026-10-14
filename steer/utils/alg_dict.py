import importlib

from ..vector_generators.sae_feature.generate_sae_feature_hparam import SaeFeatureHyperParams
from ..vector_generators.sta.generate_sta_hparam import STAHyperParams
from ..vector_generators.vector_prompt.generate_vector_prompt_hparam import VectorPromptHyperParams
from ..vector_generators.caa.generate_caa_hparam import CAAHyperParams
from ..vector_generators.lm_steer.generate_lm_steer_hparam import LmSteerHyperParams
from ..vector_generators.merge.generate_merge_vector_hparams import MergeVectorHyperParams
from ..vector_generators.reps.generate_reps_hparams import RePSHyperParams
from ..vector_generators.sft.generate_sft_hparams import SFTHyperParams
from ..vector_generators.spilt.generate_spilt_hparams import SpiltHyperParams

from ..vector_appliers.sae_feature.apply_sae_feature_hparam import ApplySaeFeatureHyperParams
from ..vector_appliers.sta.apply_sta_hparam import ApplySTAHyperParams
from ..vector_appliers.caa.apply_caa_hparam import ApplyCAAHyperParams
from ..vector_appliers.reps.apply_reps_hparam import ApplyRepsHyperParams
from ..vector_appliers.vector_prompt.apply_vector_prompt_hparam import ApplyVectorPromptHyperParams
from ..vector_appliers.lm_steer.apply_lm_steer_hparam import ApplyLmSteerHyperParams
from ..vector_appliers.prompt.apply_prompt_hparam import ApplyPromptHyperParams
from ..vector_appliers.merge.apply_merge_vector_hparam import ApplyMergeVectorHyperParams
from ..vector_appliers.sft.apply_sft_hparam import ApplySFTHyperParams
from ..vector_appliers.spilt.apply_spilt_hparam import ApplySpiltHyperParams

from ..vector_generators.lm_steer.generate_lm_steer_delta import generate_lm_steer_delta
from ..vector_generators.caa.generate_caa_vectors import generate_caa_vectors
from ..vector_generators.vector_prompt.generate_vector_prompt_vectors import generate_vector_prompt_vectors
from ..vector_generators.merge.generate_merge_vectors import generate_merge_vector
from ..vector_generators.reps.generate_reps import generate_reps
from ..vector_generators.sft.generate_sft import generate_sft
from ..vector_generators.spilt.generate_spilt import generate_spilt

from ..vector_appliers.lm_steer.apply_lm_steer import apply_lm_steer
from ..vector_appliers.caa.apply_caa import apply_caa
from ..vector_appliers.reps.apply_reps import apply_reps
from ..vector_appliers.vector_prompt.apply_vector_prompt import apply_vector_prompt
from ..vector_appliers.sae_feature.apply_sae_feature import apply_sae_feature
from ..vector_appliers.sta.apply_sta import apply_sta
from ..vector_appliers.prompt.apply_prompt import apply_prompt
from ..vector_appliers.merge.apply_merge_vector import apply_merge_vector
from ..vector_appliers.sft.apply_sft import apply_sft
from ..vector_appliers.spilt.apply_spilt import apply_spilt


def _lazy(module_path, attr):
    def _call(*args, **kwargs):
        fn = getattr(importlib.import_module(module_path), attr)
        return fn(*args, **kwargs)
    return _call


generate_sae_feature_vectors = _lazy(
    "steer.vector_generators.sae_feature.generate_sae_feature_vectors",
    "generate_sae_feature_vectors",
)
generate_sta_vectors = _lazy(
    "steer.vector_generators.sta.generate_sta_vectors",
    "generate_sta_vectors",
)

import torch

DTYPES_DICT ={
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    'float64': torch.float64,
    "bf16": torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64
}
HYPERPARAMS_CLASS_DICT = {
    'lm_steer': {'train': LmSteerHyperParams, 'apply': ApplyLmSteerHyperParams},
    'sae_feature': {'train': SaeFeatureHyperParams, 'apply': ApplySaeFeatureHyperParams},
    'sta': {'train': STAHyperParams, 'apply': ApplySTAHyperParams},
    'vector_prompt': {'train': VectorPromptHyperParams, 'apply': ApplyVectorPromptHyperParams},
    'caa': {'train': CAAHyperParams, 'apply': ApplyCAAHyperParams},
    'prompt': {'apply': ApplyPromptHyperParams},
    'merge_vector': {'train': MergeVectorHyperParams, 'apply': ApplyMergeVectorHyperParams},
    'reps':{'train': RePSHyperParams, 'apply': ApplyRepsHyperParams},
    'sft':{'train': SFTHyperParams, 'apply': ApplySFTHyperParams},
    'spilt':{'train': SpiltHyperParams, 'apply': ApplySpiltHyperParams}
}

METHODS_CLASS_DICT = {
    'lm_steer': {'train': generate_lm_steer_delta, 'apply': apply_lm_steer},
    'caa': {'train': generate_caa_vectors, 'apply': apply_caa},
    'vector_prompt': {'train': generate_vector_prompt_vectors, 'apply': apply_vector_prompt},
    'sae_feature': {'train': generate_sae_feature_vectors, 'apply': apply_sae_feature},
    'sta': {'train': generate_sta_vectors, 'apply': apply_sta},
    'prompt': {'apply': apply_prompt},
    'merge_vector': {'train': generate_merge_vector, 'apply': apply_merge_vector},
    'reps': {'train': generate_reps, 'apply': apply_reps},
    'sft': {'train': generate_sft, 'apply': apply_sft},
    'spilt': {'train': generate_spilt, 'apply': apply_spilt}
}

# Specified the methods that are supported by vLLM. This is used to filter out unsupported methods when using vLLM for inference.
VLLM_SUPPORTED_METHODS = ['caa', 'vector_prompt', 'sta', 'sae_feature']