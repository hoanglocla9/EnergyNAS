
from transformers.models.mamba2.modeling_mamba2 import Mamba2RMSNorm, Mamba2Block, Mamba2PreTrainedModel, Mamba2Output
from nni.nas.nn.extras.llm_layers import MHABlock, HybridMamba2Config, Mamba2AttentionModel, HybridMamba2Cache
import nni.retiarii.nn.pytorch as nn
import torch
from typing import Optional, Union, Tuple

@nni.retiarii.model_wrapper
class HybridSearchSpace(Mamba2PreTrainedModel): ## from Mamba2AttentionModel

    def __init__(self, 
            vocab_size=50277, 
            config=None
        ):
        super().__init__(config)
        hidden_size = nn.ValueChoice(range(128, 1025, 128), label="hidden_size")
        num_hidden_layers = nn.ValueChoice(range(8, 24, 4), label="num_hidden_layers")
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        tmp_layers = []
        num_heads=24
        head_dim=64
        state_size=128
        layer_norm_epsilon=1e-5
        pad_token_id=1
        bos_token_id=0
        eos_token_id=2
        expand=2
        conv_kernel=4
        n_groups=1
        use_bias=False
        use_conv_bias=True
        hidden_act="silu"
        initializer_range=0.1
        residual_in_fp32=True
        time_step_rank="auto"
        time_step_min=0.001
        time_step_max=0.1
        time_step_floor=1e-4
        time_step_limit=(0.0, float("inf"))
        rescale_prenorm_residual=False
        use_cache=False
        rms_norm=True
        chunk_size=256
        tie_word_embeddings=True

        tmp_layers = nn.Repeat(lambda idx: nn.LayerChoice([
                MHABlock(num_heads=32, head_dim=128, casual=True, embed_dim=hidden_size, d_conv=4, layer_idx=idx), # conv_kernel
                Mamba2Block(num_heads=num_heads, head_dim=head_dim, state_size=state_size, layer_idx=idx)
            ]), num_hidden_layers, label="n_mlpblocks")
       
        self.layers = nn.ModuleList(tmp_layers) # [Mamba2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        self.gradient_checkpointing = False
        self.norm_f = Mamba2RMSNorm(hidden_size, eps=layer_norm_epsilon) # fixed: layer_norm_epsilon
        
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridMamba2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Mamba2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = HybridMamba2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)

            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return Mamba2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
    

    
config = HybridMamba2Config(num_heads=24, ## num_heads = hidden_size * expand / head_dim
        head_dim=64,
        vocab_size=50277,
        hidden_size=768,
        state_size=128,
        num_hidden_layers=8,
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        expand=2,
        conv_kernel=4,
        n_groups=1,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual=False,
        use_cache=False,
        rms_norm=True,
        chunk_size=256,
        tie_word_embeddings=True,
        attn_layer_idxs=[6, 10],
        attn_params = {'num_heads': 32,  'head_dim':128, 'causal': True} # 'num_heads_kv': 24, 'mlp_dim': 32, 
)

search_space = HybridSearchSpace(config)