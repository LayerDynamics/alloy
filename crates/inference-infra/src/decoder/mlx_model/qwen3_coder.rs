use super::*;
use crate::decoder::{
    loader::{
        CheckpointTensor, CheckpointVerification, LoadOptions, LoadedTensorSet,
        load_tensor_set_with_packing,
    },
    qwen3_coder::{CoderConfig, invalid},
};
use std::path::Path;

struct ExpertProjection {
    packed: Array,
    scales: Array,
    biases: Array,
}

impl ExpertProjection {
    fn upload(
        matrix: MatrixWeight,
        experts: usize,
        output: usize,
        input: usize,
    ) -> MlxDecoderResult<Self> {
        let MatrixWeight::Affine4 {
            packed,
            scales,
            biases,
            rows,
            cols,
            group_size,
        } = matrix
        else {
            return Err(invalid("expected 4-bit expert bank").into());
        };
        if rows != experts * output || cols != input || group_size != 64 {
            return Err(invalid("incorrect expert bank dimensions").into());
        }
        let groups = [experts, output, input / group_size];
        Ok(Self {
            packed: upload_u32(packed, &[experts, output, input / 8])?,
            scales: upload_f32(scales, &groups)?,
            biases: upload_f32(biases, &groups)?,
        })
    }

    fn view(&self) -> ops::AffineExperts<'_> {
        ops::AffineExperts {
            packed: &self.packed,
            scales: &self.scales,
            biases: &self.biases,
            bits: 4,
            group_size: 64,
        }
    }
}

pub(super) struct RoutedLayer {
    router: MlxLinear,
    gate: ExpertProjection,
    up: ExpertProjection,
    down: ExpertProjection,
    top_k: usize,
}

impl RoutedLayer {
    pub(super) fn apply(&self, x: &Array, stream: &Stream) -> MlxDecoderResult<Array> {
        let logits = self.router.apply(x, stream)?;
        let banks = ops::SwitchGluExperts {
            gate: self.gate.view(),
            up: self.up.view(),
            down: self.down.view(),
        };
        Ok(ops::routed_swiglu(
            x, &logits, &banks, self.top_k, true, stream,
        )?)
    }
    pub(super) fn visit<'a>(&'a self, visitor: &mut impl FnMut(&'a Array)) {
        self.router.visit(visitor);
        for bank in [&self.gate, &self.up, &self.down] {
            for tensor in [&bank.packed, &bank.scales, &bank.biases] {
                visitor(tensor);
            }
        }
    }
}

struct Upload(LoadedTensorSet);
impl Upload {
    fn matrix(&mut self, name: &str) -> MlxDecoderResult<MatrixWeight> {
        match self.0.remove(name) {
            Some(CheckpointTensor::Matrix(w)) => Ok(w),
            _ => Err(invalid("missing validated Qwen3-Coder matrix").into()),
        }
    }
    fn linear(&mut self, name: &str, rows: usize, cols: usize) -> MlxDecoderResult<MlxLinear> {
        MlxLinear::upload(self.matrix(name)?, rows, cols)
    }
    fn norm(&mut self, name: &str, width: usize) -> MlxDecoderResult<Array> {
        match self.0.remove(name) {
            Some(CheckpointTensor::Float { values, .. }) => upload_f32(values, &[width]),
            _ => Err(invalid("missing validated Qwen3-Coder norm").into()),
        }
    }

    fn layer(&mut self, config: &CoderConfig, index: usize) -> MlxDecoderResult<MlxLayer> {
        let c = &config.decoder;
        let p = format!("model.layers.{index}");
        let h = c.hidden_size;
        let m = c.intermediate_size;
        let e = config.experts;
        let mut expert = |name, rows, cols| {
            ExpertProjection::upload(
                self.matrix(&format!("{p}.mlp.switch_mlp.{name}.weight"))?,
                e,
                rows,
                cols,
            )
        };
        let gate = expert("gate_proj", m, h)?;
        let up = expert("up_proj", m, h)?;
        let down = expert("down_proj", h, m)?;
        let routed = RoutedLayer {
            router: self.linear(&format!("{p}.mlp.gate.weight"), e, h)?,
            gate,
            up,
            down,
            top_k: config.top_k,
        };
        Ok(MlxLayer {
            input_layernorm: self.norm(&format!("{p}.input_layernorm.weight"), h)?,
            q_proj: self.linear(&format!("{p}.self_attn.q_proj.weight"), c.q_dim(), h)?,
            k_proj: self.linear(&format!("{p}.self_attn.k_proj.weight"), c.kv_dim(), h)?,
            v_proj: self.linear(&format!("{p}.self_attn.v_proj.weight"), c.kv_dim(), h)?,
            o_proj: self.linear(&format!("{p}.self_attn.o_proj.weight"), h, c.q_dim())?,
            q_norm: Some(self.norm(&format!("{p}.self_attn.q_norm.weight"), c.head_dim)?),
            k_norm: Some(self.norm(&format!("{p}.self_attn.k_norm.weight"), c.head_dim)?),
            post_attention_layernorm: self
                .norm(&format!("{p}.post_attention_layernorm.weight"), h)?,
            mlp: FeedForward::Routed(Box::new(routed)),
        })
    }
}

impl MlxDecoderModel {
    /// Load the complete trained, mixed-precision MoE through the bounded shared
    /// reader. Every consumed shard must match the trusted checkpoint manifest.
    pub fn load_qwen3_coder_verified(
        directory: &Path,
        options: LoadOptions,
        verification: &CheckpointVerification<'_>,
        stream: &Stream,
    ) -> MlxDecoderResult<Self> {
        let (config, tensors) = load_tensor_set_with_packing(
            directory,
            options,
            verification,
            CoderConfig::inventory,
            |_| false,
        )?;
        let payload = tensors.values().try_fold(0u64, |sum, t| {
            let bytes = match t {
                CheckpointTensor::Float { values, .. } => values.len() * 4,
                CheckpointTensor::Matrix(w) => w.resident_bytes(),
            };
            sum.checked_add(bytes as u64)
                .ok_or(ModelError::DimensionOverflow("routed payload"))
        })?;
        let result = (|| {
            let c = &config.decoder;
            let mut upload = Upload(tensors);
            let mut layers = Vec::with_capacity(c.num_layers);
            for i in 0..c.num_layers {
                if (verification.cancelled)() {
                    return Err(ModelError::Cancelled.into());
                }
                layers.push(upload.layer(&config, i)?);
            }
            let native = MlxWeights {
                layers,
                embedding: upload.linear(
                    "model.embed_tokens.weight",
                    c.vocab_size,
                    c.hidden_size,
                )?,
                final_layernorm: upload.norm("model.norm.weight", c.hidden_size)?,
                lm_head: Some(upload.linear("lm_head.weight", c.vocab_size, c.hidden_size)?),
            };
            if !upload.0.is_empty() {
                return Err(invalid("unconsumed Qwen3-Coder tensor").into());
            }
            if (verification.cancelled)() {
                return Err(ModelError::Cancelled.into());
            }
            Self::from_uploaded(
                config.decoder,
                native,
                payload,
                options.max_weight_bytes as u64,
            )
        })();
        complete_stream(result, stream)
    }
}
