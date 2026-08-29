# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
REIGN model configuration.

TODO: Refactor docstrings for REIGN.
"""
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ReignConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ReignModel`]. It is used to
    instantiate a REIGN model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the REIGN
    `base` architecture (see [`ReignBaseConfig`]).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pooling_strategy (`str`, *optional*, defaults to `"mean"`):
            The strategy to pool the model's hidden states.
        torch_dtype (`str`, *optional*, defaults to `"float16"`):
            The data type of the model.
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        is_decoder (`bool`, *optional*, defaults to `False`): Whether the model is used as a decoder or not. Only
        gn_chunk_size (`int`, *optional*, defaults to 512):
            Number of tokens per chunk fed to the Guidance Network when this REIGN checkpoint was trained.
            Stored as metadata of the extraction pipeline; it does not affect the model's forward math
            (REIGN is permutation-equivariant over chunk embeddings) but it is load-bearing for reproducing
            the chunk embeddings the encoder expects at inference time.
        position_embedding_type (`str`, *optional*, defaults to `None`):
            How chunk position enters the encoder. The default `None` adds
            nothing, leaving REIGN a permutation-equivariant set function over
            chunk embeddings -- the published design, and the behaviour of every
            checkpoint trained before this option existed. `"absolute"` adds a
            learned per-position embedding and `"sinusoidal"` a fixed one; these
            exist so the design choice can be tested empirically rather than
            asserted. Any other value (including legacy leftovers such as
            `"relative_key"`) is stored but inert. Positions beyond
            `max_position_embeddings` are clamped to the last index rather than
            truncated, so documents longer than the table still encode.
        max_position_embeddings (`int`, *optional*, defaults to `None`):
            Size of the chunk-position table, used only when
            `position_embedding_type` is `"absolute"` or `"sinusoidal"`; falls
            back to 4096 positions (~2M tokens at 512-token chunks) when unset.
        gn_stride (`int`, *optional*, defaults to 384):
            Stride between successive chunks fed to the Guidance Network when this REIGN checkpoint was
            trained. ``gn_stride == gn_chunk_size`` gives non-overlapping chunks; ``gn_stride < gn_chunk_size``
            gives sliding-window chunks sharing ``gn_chunk_size - gn_stride`` tokens between neighbours.
            Same metadata-only role as ``gn_chunk_size``.

    Examples:

    ```python
    >>> from reign import ReignConfig, ReignModel

    >>> # Initializing a REIGN reign-base style configuration
    >>> configuration = ReignConfig()

    >>> # Initializing a model (with random weights) from the bert-base-uncased style configuration
    >>> model = ReignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "reign"

    def __init__(
        self,
        vocab_size=3,
        gn_projection_dim=1024,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        pooling_strategy="mean",
        torch_dtype="float32",
        use_special_tokens=False,
        is_decoder=False,
        gn_chunk_size=512,
        gn_stride=384,
        position_embedding_type=None,
        max_position_embeddings=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.gn_projection_dim = gn_projection_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pooling_strategy = pooling_strategy
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_special_tokens = use_special_tokens
        self.is_decoder = is_decoder
        # Metadata of the GN extraction pipeline used to produce chunk embeddings
        # for this checkpoint. Does not affect forward math.
        self.gn_chunk_size = gn_chunk_size
        self.gn_stride = gn_stride
        # Chunk-position signal. Stored verbatim and NOT validated here: older
        # configs persist arbitrary leftovers of the removed positional surface
        # (None, "relative_key", ...) and must keep loading untouched. Only
        # "absolute" and "sinusoidal" activate anything in ReignEmbeddings;
        # every other value stays inert, which is the historical behaviour.
        # The CLI (`--position-embedding-type`) constrains the ablation's choices.
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings


class ReignSmallConfig(ReignConfig):
    r"""
    This is a smaller variant of the REIGN configuration designed for resource-constrained environments
    or faster training/inference. It reduces the model size while maintaining the core architecture.

    The key differences from the base ReignConfig are:
    - `hidden_size`: 384 (reduced from 1024)
    - `intermediate_size`: 1024 (reduced from 4096)
    - `num_hidden_layers`: 6 (reduced from 12)
    - `num_attention_heads`: 6 (adjusted to maintain head_dim=64)

    This configuration provides approximately 4x reduction in model parameters while preserving
    the fundamental capabilities of the REIGN architecture.

    Examples:

    ```python
    >>> from reign import ReignSmallConfig, ReignModel

    >>> # Initializing a REIGN small configuration
    >>> configuration = ReignSmallConfig()

    >>> # Initializing a smaller model from the configuration
    >>> model = ReignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,  # Adjusted to maintain head_dim=64 (384/6=64)
        intermediate_size: int = 1536,
        gn_projection_dim: int = 384,  # Adjusted to match hidden_size
        **kwargs,
    ):
        """
        Initialize ReignSmallConfig with smaller model dimensions.

        Args:
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 384.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 6.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 6.
            intermediate_size (int, optional): Dimensionality of the feed-forward layer. Defaults to 1024.
            gn_projection_dim (int, optional): Projection dimension. Defaults to 384.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignBaseConfig(ReignConfig):
    r"""
    This is the base variant of the REIGN configuration following standard transformer "base" model sizes.
    It provides a good balance between model capacity and computational efficiency.

    The key parameters for the base configuration are:
    - `hidden_size`: 768 (standard base model size)
    - `intermediate_size`: 3072 (4x hidden_size, standard ratio)
    - `num_hidden_layers`: 12 (standard base model depth)
    - `num_attention_heads`: 12 (maintains head_dim=64)

    This configuration is suitable for most production use cases and provides good performance
    while maintaining reasonable computational requirements.

    Examples:

    ```python
    >>> from reign import ReignBaseConfig, ReignModel

    >>> # Initializing a REIGN base configuration
    >>> configuration = ReignBaseConfig()

    >>> # Initializing a base model from the configuration
    >>> model = ReignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,  # Maintains head_dim=64 (768/12=64)
        intermediate_size: int = 3072,  # 4x hidden_size
        gn_projection_dim: int = 768,  # Matched to hidden_size
        **kwargs,
    ):
        """
        Initialize ReignBaseConfig with standard base model dimensions.

        Args:
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 768.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 12.
            intermediate_size (int, optional): Dimensionality of the feed-forward layer. Defaults to 3072.
            gn_projection_dim (int, optional): Projection dimension. Defaults to 768.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignLargeConfig(ReignConfig):
    r"""
    This is the large variant of the REIGN configuration designed for maximum model capacity
    and performance. It significantly increases the model size for demanding applications.

    The key parameters for the large configuration are:
    - `hidden_size`: 1024 (large model size)
    - `intermediate_size`: 4096 (4x hidden_size, standard ratio)
    - `num_hidden_layers`: 24 (doubled depth for increased capacity)
    - `num_attention_heads`: 16 (maintains head_dim=64)

    This configuration provides maximum model capacity for high-performance requirements
    but requires significant computational resources for training and inference.

    Examples:

    ```python
    >>> from reign import ReignLargeConfig, ReignModel

    >>> # Initializing a REIGN large configuration
    >>> configuration = ReignLargeConfig()

    >>> # Initializing a large model from the configuration
    >>> model = ReignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,  # Doubled depth for increased capacity
        num_attention_heads: int = 16,  # Maintains head_dim=64 (1024/16=64)
        intermediate_size: int = 4096,  # 4x hidden_size
        gn_projection_dim: int = 1024,  # Matched to hidden_size
        **kwargs,
    ):
        """
        Initialize ReignLargeConfig with large model dimensions.

        Args:
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 1024.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 24.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 16.
            intermediate_size (int, optional): Dimensionality of the feed-forward layer. Defaults to 4096.
            gn_projection_dim (int, optional): Projection dimension. Defaults to 1024.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignXLargeConfig(ReignConfig):
    r"""
    This is the extra large variant of the REIGN configuration designed for maximum model capacity
    and performance. It significantly increases the model size for demanding applications.

    The key parameters for the extra large configuration are:
    - `hidden_size`: 1024 (extra large model size)
    - `intermediate_size`: 5120 (5x hidden_size for increased capacity)
    - `num_hidden_layers`: 24 (maximum depth for increased capacity)
    - `num_attention_heads`: 16 (maintains head_dim=64)

    This configuration provides maximum model capacity for high-performance requirements
    but requires significant computational resources for training and inference.

    Examples:

    ```python
    >>> from reign import ReignXLargeConfig, ReignModel

    >>> # Initializing a REIGN extra large configuration
    >>> configuration = ReignXLargeConfig()

    >>> # Initializing an extra large model from the configuration
    >>> model = ReignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,  # Maximum depth for increased capacity
        num_attention_heads: int = 16,  # Maintains head_dim=64 (1024/16=64)
        intermediate_size: int = 5120,  # 5x hidden_size for increased capacity
        gn_projection_dim: int = 1024,  # Matched to hidden_size
        **kwargs,
    ):
        """
        Initialize ReignXLargeConfig with extra large model dimensions.

        Args:
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 1024.
            num_hidden_layers (int, optional): Number of hidden layers. Defaults to 24.
            num_attention_heads (int, optional): Number of attention heads. Defaults to 16.
            intermediate_size (int, optional): Dimensionality of the feed-forward layer. Defaults to 5120.
            gn_projection_dim (int, optional): Projection dimension. Defaults to 1024.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


# ================================
# New Layer-Based Config Variants
# ================================


class ReignTinyL1Config(ReignConfig):
    r"""
    Tiny variant of REIGN configuration with 1 layer for minimal resource usage.

    Key parameters:
    - `hidden_size`: 192 (minimal size)
    - `num_hidden_layers`: 1 (single transformer layer)
    - `num_attention_heads`: 3 (maintains head_dim=64)
    - `intermediate_size`: 768 (4x hidden_size)
    - `gn_projection_dim`: 384 (matches GTE-small)
    """

    def __init__(
        self,
        hidden_size: int = 192,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 3,  # 192/3=64 head_dim
        intermediate_size: int = 768,  # 4x hidden_size
        gn_projection_dim: int = 384,  # GTE-small projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignTinyL3Config(ReignConfig):
    r"""
    Tiny variant of REIGN configuration with 3 layers for lightweight usage.

    Key parameters:
    - `hidden_size`: 192 (minimal size)
    - `num_hidden_layers`: 3 (three transformer layers)
    - `num_attention_heads`: 3 (maintains head_dim=64)
    - `intermediate_size`: 768 (4x hidden_size)
    - `gn_projection_dim`: 384 (matches GTE-small)
    """

    def __init__(
        self,
        hidden_size: int = 192,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 3,  # 192/3=64 head_dim
        intermediate_size: int = 768,  # 4x hidden_size
        gn_projection_dim: int = 384,  # GTE-small projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignSmallL2Config(ReignConfig):
    r"""
    Small variant of REIGN configuration with 2 layers.

    Key parameters:
    - `hidden_size`: 384 (small size)
    - `num_hidden_layers`: 2 (two transformer layers)
    - `num_attention_heads`: 6 (maintains head_dim=64)
    - `intermediate_size`: 1536 (4x hidden_size)
    - `gn_projection_dim`: 384 (matches GTE-small)
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 6,  # 384/6=64 head_dim
        intermediate_size: int = 1536,  # 4x hidden_size
        gn_projection_dim: int = 384,  # GTE-small projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignSmallL3Config(ReignConfig):
    r"""
    Small variant of REIGN configuration with 3 layers.

    Key parameters:
    - `hidden_size`: 384 (small size)
    - `num_hidden_layers`: 3 (three transformer layers)
    - `num_attention_heads`: 6 (maintains head_dim=64)
    - `intermediate_size`: 1536 (4x hidden_size)
    - `gn_projection_dim`: 384 (matches GTE-small)
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 6,  # 384/6=64 head_dim
        intermediate_size: int = 1536,  # 4x hidden_size
        gn_projection_dim: int = 384,  # GTE-small projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignBaseL3Config(ReignConfig):
    r"""
    Base variant of REIGN configuration with 3 layers.

    Key parameters:
    - `hidden_size`: 768 (base size)
    - `num_hidden_layers`: 3 (three transformer layers)
    - `num_attention_heads`: 12 (maintains head_dim=64)
    - `intermediate_size`: 3072 (4x hidden_size)
    - `gn_projection_dim`: 768 (matches GTE-base)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,  # 768/12=64 head_dim
        intermediate_size: int = 3072,  # 4x hidden_size
        gn_projection_dim: int = 768,  # GTE-base projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignBaseL4Config(ReignConfig):
    r"""
    Base variant of REIGN configuration with 4 layers.

    Key parameters:
    - `hidden_size`: 768 (base size)
    - `num_hidden_layers`: 4 (four transformer layers)
    - `num_attention_heads`: 12 (maintains head_dim=64)
    - `intermediate_size`: 3072 (4x hidden_size)
    - `gn_projection_dim`: 768 (matches GTE-base)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 12,  # 768/12=64 head_dim
        intermediate_size: int = 3072,  # 4x hidden_size
        gn_projection_dim: int = 768,  # GTE-base projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignLargeL4Config(ReignConfig):
    r"""
    Large variant of REIGN configuration with 4 layers.

    Key parameters:
    - `hidden_size`: 1024 (large size)
    - `num_hidden_layers`: 4 (four transformer layers)
    - `num_attention_heads`: 16 (maintains head_dim=64)
    - `intermediate_size`: 4096 (4x hidden_size)
    - `gn_projection_dim`: 1024 (matches GTE-large)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 16,  # 1024/16=64 head_dim
        intermediate_size: int = 4096,  # 4x hidden_size
        gn_projection_dim: int = 1024,  # GTE-large projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignLargeL6Config(ReignConfig):
    r"""
    Large variant of REIGN configuration with 6 layers (maximum depth).

    Key parameters:
    - `hidden_size`: 1024 (large size)
    - `num_hidden_layers`: 6 (six transformer layers - maximum)
    - `num_attention_heads`: 16 (maintains head_dim=64)
    - `intermediate_size`: 4096 (4x hidden_size)
    - `gn_projection_dim`: 1024 (matches GTE-large)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,  # 1024/16=64 head_dim
        intermediate_size: int = 4096,  # 4x hidden_size
        gn_projection_dim: int = 1024,  # GTE-large projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignXLargeL4Config(ReignConfig):
    r"""
    Extra Large variant of REIGN configuration with 4 layers and increased capacity.

    Key parameters:
    - `hidden_size`: 1024 (extra large size)
    - `num_hidden_layers`: 4 (four transformer layers)
    - `num_attention_heads`: 16 (maintains head_dim=64)
    - `intermediate_size`: 5120 (5x hidden_size for increased capacity)
    - `gn_projection_dim`: 1024 (matches GTE-large)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 16,  # 1024/16=64 head_dim
        intermediate_size: int = 5120,  # 5x hidden_size for increased capacity
        gn_projection_dim: int = 1024,  # GTE-large projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )


class ReignXLargeL6Config(ReignConfig):
    r"""
    Extra Large variant of REIGN configuration with 6 layers (maximum depth) and increased capacity.

    Key parameters:
    - `hidden_size`: 1024 (extra large size)
    - `num_hidden_layers`: 6 (six transformer layers - maximum)
    - `num_attention_heads`: 16 (maintains head_dim=64)
    - `intermediate_size`: 5120 (5x hidden_size for increased capacity)
    - `gn_projection_dim`: 1024 (matches GTE-large)
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 16,  # 1024/16=64 head_dim
        intermediate_size: int = 5120,  # 5x hidden_size for increased capacity
        gn_projection_dim: int = 1024,  # GTE-large projection
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            gn_projection_dim=gn_projection_dim,
            **kwargs,
        )
