#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::LayerNorm;
use mistralrs_quant::{
    ColumnParallelLayer, NonZeroOp, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use std::{collections::HashMap, sync::Arc};

use candle_nn::ops::softmax_last_dim;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    kv_cache::{EitherCache, KvCache, NormalCache}, 
    layers::{
        self, layer_norm, Activation, CausalMasker, MatMul, RotaryEmbedding,
    },
    layers_masker::{masked_fill, PastKvLenCache},
    moe::{MoEExperts, MoEExpertsConfig},
    ops::TopKLastDimOp,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        NormalLoadingMetadata, NormalModel,
    },
    utils::{
        progress::NiceProgressBar,
        unvarbuilder::UnVarBuilder,
    },
};
use crate::serde_default_fn;

serde_default_fn!(bool, word_emb_default, false);

/// MiMo-V2-Flash 模型配置
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) sliding_window: Option<usize>,
    
    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) lm_head_bias: bool,
    pub(crate) attention_bias: bool,
    
    // MOE specific fields
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) router_jitter_noise: f64,
    
    #[serde(default = "word_emb_default")]
    pub(crate) tie_word_embeddings: bool,
}



impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        
        let q_proj = Arc::new(layers::Linear::new(
            vb.pp("q_proj"),
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            false,
        )?) as Arc<dyn QuantMethod>;
        
        let k_proj = Arc::new(layers::Linear::new(
            vb.pp("k_proj"),
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            false,
        )?) as Arc<dyn QuantMethod>;
        
        let v_proj = Arc::new(layers::Linear::new(
            vb.pp("v_proj"),
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            false,
        )?) as Arc<dyn QuantMethod>;
        
        let o_proj = Arc::new(layers::Linear::new(
            vb.pp("o_proj"),
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            false,
        )?) as Arc<dyn QuantMethod>;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams::default(),
        })
    }
    
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        _kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        
        // Query, Key, Value projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        // Reshape for attention
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        
        // Apply rotary embedding
        let q = self.rotary_emb.forward(&q, position_ids)?;
        let k = self.rotary_emb.forward(&k, position_ids)?;
        
        // Perform attention - always use standard implementation for now
        let attn_weights = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)?;
        let attn_weights = attn_weights / (self.head_dim as f64).sqrt();
        let attn_weights = if let Some(mask) = mask {
            masked_fill(&attn_weights, mask, f64::NEG_INFINITY)?
        } else {
            attn_weights
        };
        let attn_weights = softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        
        // Reshape and output projection
        let attn_output = attn_output.reshape((b_sz, seq_len, -1))?;
        self.o_proj.forward(&attn_output)
    }
}

struct MoeMlp {
    experts: MoEExperts,
    gate: Arc<dyn QuantMethod>,
    act: Activation,
}

impl MoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let moe_config = MoEExpertsConfig {
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
        };
        
        let experts = MoEExperts::new(
            &moe_config,
            vb.pp("experts"),
            layer_device,
            comm,
            false,
            &cfg.quantization_config,
            cfg.hidden_act.clone(),
        )?;
        
        let gate = Arc::new(layers::Linear::new(
            vb.pp("gate_proj"),
            cfg.hidden_size,
            cfg.num_experts,
            false,
        )?) as Arc<dyn QuantMethod>;
        
        Ok(Self {
            experts,
            gate,
            act: cfg.hidden_act.clone(),
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute gate logits
        let gate_logits = self.gate.forward(x)?;
        
        // Compute top-k experts and weights
        let k = self.experts.config.num_experts_per_tok;
        let TopKLastDimOp::TopKOutput(topk_weights, topk_ids) = gate_logits.topk(k)?;
        
        // Apply MOE experts
        let moe_output = self.experts.forward(x, topk_weights, &topk_ids, false)?;
        
        Ok(moe_output)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb.clone(),
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
            comm,
        )?;
        
        let mlp = MoeMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
            mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device),
            comm,
        )?;
        
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
    
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
        _kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            position_ids,
            None,
            metadata,
            flash_params,
        )?;
        
        let xs = (xs + residual)?;
        let residual = &xs;
        
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Arc<dyn QuantMethod>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: Option<usize>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let comm = Arc::new(mistralrs_quant::Comm::from_device(
            mistralrs_quant::Id::new(),
            &device,
            0,
            1,
        )?);
        
        let mapper = crate::device_map::DummyDeviceMapper { nm_device: device.clone() };
        
        // Embedding layer
        let embed_tokens = candle_nn::Embedding::new(
            vb.pp("embed_tokens"),
            cfg.vocab_size,
            cfg.hidden_size,
        )?;
        
        // Rotary embedding
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            cfg.head_dim(),
            cfg.max_position_embeddings,
            &device,
            false,
            candle_core::DType::F32,
        )?);
        
        // Paged attention setup
        let paged_attn = None;
        
        // Create decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb.pp(&format!("layers.{layer_idx}")),
                &mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn.clone(),
                device.clone(),
                &comm,
            )?;
            layers.push(layer);
        }
        
        // Final layer norm
        let norm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("norm"),
        )?;
        
        // LM head
        let lm_head = layers::Linear::new(
            vb.pp("lm_head"),
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.lm_head_bias,
        )?;
        let lm_head = Arc::new(lm_head) as Arc<dyn QuantMethod>;
        
        // Cache setup
        let cache = EitherCache::Normal(NormalCache::new(cfg.num_hidden_layers, cfg.max_position_embeddings));
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
            cache,
            max_seq_len: cfg.max_position_embeddings,
            mapper: Box::new(mapper),
            sliding_window: cfg.sliding_window,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
            },
        })
    }
}

impl NormalModel for Model {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        cache_pos: Option<&Tensor>,
        flash_params: &FlashParams,
    ) -> Result<(Tensor, Vec<Option<Tensor>>)> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        
        // Embedding lookup
        let mut x = self.embed_tokens.forward(input_ids)?;
        
        // Position ids setup
        let position_ids = if let Some(cache_pos) = cache_pos {
            cache_pos
        } else {
            &Tensor::arange(0u32, seq_len as u32, &self.device)?
        };
        
        // Convert position_ids to vector of usize
        let position_ids_vec = position_ids.flatten_all()?
            .to_dtype(candle_core::DType::I32)?
            .i32s()?
            .into_iter()
            .map(|p| p as usize)
            .collect::<Vec<_>>();
        
        // Seqlen offsets
        let seqlen_offsets = (0..b_sz as usize)
            .map(|i| i * seq_len)
            .collect::<Vec<_>>();
        
        // Mask setup
        let mask = attention_mask;
        
        // Process each decoder layer
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Get KV cache for this layer
            let mut kv_cache = None;
            if let EitherCache::Normal(normal_cache) = &self.cache {
                let mut normal_cache = normal_cache.lock().unwrap();
                if let Some(cache) = normal_cache.0.get_mut(layer_idx) {
                    kv_cache = Some(cache);
                }
            }
            
            let layer_output = layer.forward(
                &x,
                mask,
                &seqlen_offsets,
                &position_ids_vec,
                kv_cache,
                None,
                flash_params,
            )?;
            x = layer_output;
        }
        
        // Final normalization and LM head
        x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        
        Ok((logits, Vec::new()))
    }
    
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    
    fn device(&self) -> &Device {
        &self.device
    }
    
    fn is_xlora(&self) -> bool {
        false
    }
    
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn crate::amoe::MlpLayer>> {
        // MOE models don't use standard MLP layers for AnyMoe
        Vec::new()
    }
    
    fn create_anymoe_layers(
        &mut self,
        _additional_vbs: Vec<ShardedVarBuilder>,
        _config: crate::amoe::AnyMoeConfig,
        _prefix_mlp: (String, String),
        _layers: Vec<usize>,
        _expert_type: crate::amoe::AnyMoeExpertType,
        _gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        // MOE layers are already created directly in the model
        Ok(())
    }
    
    fn amoe_supported(&self) -> bool {
        true
    }
}


