// Native hybrid execution shares engine policy but never the transformer model.
use inference_infra::decoder::qwen35::{
    config::Qwen35Config,
    model::{ForwardOptions as HybridForwardOptions, MlxQwen35Model, Qwen35Cache},
};

impl Qwen3Backend {
    fn load_hybrid(
        &self,
        request: ModelLoadRequest<'_>,
        data: &ArtifactData,
        config: &Qwen35Config,
    ) -> Result<Box<dyn LoadedModel>, EngineError> {
        let control = request.control();
        control.check_cancelled()?;
        let peak = loading_peak_bound(data, &self.limits)?;
        if request.limits().max_source_bytes() < data.source_bytes
            || request.limits().max_resident_bytes() < data.resident_bound
            || request.limits().max_load_peak_bytes() < peak
        {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "native hybrid load overlap",
                required_bytes: peak,
                available_bytes: request.limits().max_load_peak_bytes(),
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        let device = Device::new(
            match self.execution {
                Qwen3Execution::MlxCpu => DeviceType::Cpu,
                Qwen3Execution::MlxMetal => DeviceType::Gpu,
            },
            0,
        )
        .map_err(|e| execution_error("create hybrid device", e.to_string()))?;
        if !device
            .is_available()
            .map_err(|e| execution_error("query hybrid device", e.to_string()))?
        {
            return Err(EngineError::unsupported(
                "hybrid execution device",
                "requested device is unavailable; no fallback",
            ));
        }
        let stream = Rc::new(
            Stream::new(&device)
                .map_err(|e| execution_error("create hybrid stream", e.to_string()))?,
        );
        let cancelled = || control.is_cancelled();
        let (loaded_config, weights) = inference_infra::decoder::qwen35::weights::load_verified(
            &data.specification.directory,
            LoadOptions {
                max_file_bytes: as_usize(data.source_bytes)?,
                max_weight_bytes: as_usize(sum(&[
                    data.decoded_bound,
                    self.limits.loader_headroom_bytes,
                ])?)?,
            },
            &CheckpointVerification {
                sha256: &data.checksums,
                cancelled: &cancelled,
            },
        )
        .map_err(|e| model_error(e, "load verified hybrid weights"))?;
        control.check_cancelled()?;
        let model = Rc::new(
            MlxQwen35Model::from_weights(
                loaded_config,
                weights,
                request.limits().max_resident_bytes(),
                &stream,
            )
            .map_err(|e| mlx_model_error(e, control, "construct native hybrid model"))?,
        );
        control.check_cancelled()?;
        let memory = BufferMemoryUsage::new(
            model
                .weight_buffer_bytes()
                .map_err(|e| mlx_model_error(e, control, "measure hybrid weights"))?,
            model.weight_payload_bytes(),
        )?;
        let mut capabilities = self.registration.capabilities().clone();
        capabilities.context_limit_tokens = capabilities.context_limit_tokens.min(config.context);
        capabilities.weight_representations =
            vec![match (config.quantization_bits, config.group_size) {
                (4, Some(group)) => WeightRepresentation::Affine4 {
                    group_size: NonZeroUsize::new(group).unwrap(),
                },
                (8, Some(group)) => WeightRepresentation::Affine8 {
                    group_size: NonZeroUsize::new(group).unwrap(),
                },
                (_, None) => WeightRepresentation::F32,
                _ => return Err(internal("unvalidated hybrid quantization")),
            }];
        Ok(Box::new(HybridLoaded {
            model,
            stream,
            artifact: request.artifact().id().clone(),
            instance: ModelInstanceId::new(),
            registration: self.registration.clone(),
            capabilities,
            memory,
            max_prefill_tokens: self.limits.prefill_chunk_tokens.get(),
        }))
    }
}

struct HybridLoaded {
    model: Rc<MlxQwen35Model>,
    stream: Rc<Stream>,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    memory: BufferMemoryUsage,
    max_prefill_tokens: usize,
}
impl LoadedModel for HybridLoaded {
    fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }
    fn artifact_id(&self) -> &ArtifactId {
        &self.artifact
    }
    fn registration(&self) -> &FamilyRegistration {
        &self.registration
    }
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        Ok(self.memory)
    }
    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}
impl Drop for HybridLoaded {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

struct HybridCache {
    inner: Qwen35Cache,
    model: Rc<MlxQwen35Model>,
    stream: Rc<Stream>,
    owner: ModelInstanceId,
    mode: CacheMode,
}
impl FamilyCache for HybridCache {
    fn owner(&self) -> &ModelInstanceId {
        &self.owner
    }
    fn representation(&self) -> &CacheRepresentationId {
        &self.mode.representation
    }
    fn mode(&self) -> &CacheMode {
        &self.mode
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn retained_tokens(&self) -> usize {
        self.inner.len()
    }
    fn capacity_tokens(&self) -> usize {
        self.inner.capacity()
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        let used = if self.inner.is_empty() {
            0
        } else {
            self.model
                .config()
                .cache_payload_plan(self.inner.len())
                .map_err(|e| execution_error("count hybrid cache payload", e.to_string()))?
                .total_overlap_bytes
                / 2
        };
        BufferMemoryUsage::new(
            self.inner
                .buffer_bytes()
                .map_err(|e| execution_error("measure hybrid cache", e.to_string()))?,
            used,
        )
    }
    fn supports_truncation(&self) -> bool {
        false
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner
            .reset(&self.model)
            .map_err(|e| execution_error("reset hybrid state", e.to_string()))
    }
    fn truncate(&mut self, _: usize) -> Result<(), EngineError> {
        Err(EngineError::unsupported(
            "hybrid truncation",
            "recurrent prefixes require reset and rebuild",
        ))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    // Full-KV-only export would omit recurrent state. Keep the trait's explicit
    // unsupported persistence default so restart rebuilds from source tokens.
}
impl Drop for HybridCache {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl HybridLoaded {
    fn forward(
        &self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        control.check_cancelled()?;
        if cache.owner() != &self.instance
            || cache.mode() != &family_cache_mode(BackendFamily::Qwen35)?
        {
            return Err(EngineError::invalid_request(
                "hybrid cache",
                "foreign model instance or representation",
            ));
        }
        let cache = cache
            .as_any_mut()
            .downcast_mut::<HybridCache>()
            .ok_or_else(|| {
                EngineError::invalid_request("hybrid cache", "wrong native cache type")
            })?;
        if !Rc::ptr_eq(&cache.model, &self.model) {
            return Err(EngineError::invalid_request(
                "hybrid cache",
                "foreign native model",
            ));
        }
        if tokens.is_empty() || tokens.len() > self.max_prefill_tokens {
            return Err(EngineError::invalid_request(
                "tokens",
                "hybrid prefill exceeds its admitted chunk",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("hybrid position overflow"))?;
        if end > cache.inner.capacity() {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.inner.capacity(),
            });
        }
        let plan = self
            .model
            .config()
            .step_payload_plan(tokens.len(), cache.inner.capacity(), false)
            .map_err(|e| mlx_model_error(e, control, "plan hybrid execution"))?;
        let result = self
            .model
            .forward(
                tokens,
                &mut cache.inner,
                HybridForwardOptions {
                    all_logits: false,
                    output_limit: plan.output_bytes,
                    work_limit: plan.work_bytes,
                },
                &self.stream,
                &|| control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "native hybrid execution"))
            .and_then(|output| TokenLogits::new(output.into_values()));
        if result.is_err() {
            let _ = cache.inner.reset(&self.model);
        }
        result
    }
}
impl AutoregressiveModel for HybridLoaded {
    fn vocabulary_size(&self) -> usize {
        self.model.config().vocabulary
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        control.check_cancelled()?;
        if request.mode() != &family_cache_mode(BackendFamily::Qwen35)? {
            return Err(EngineError::unsupported(
                "hybrid cache mode",
                "requires full KV and recurrent state together",
            ));
        }
        if request.capacity_tokens() > self.capabilities.context_limit_tokens {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: request.capacity_tokens(),
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: self.capabilities.context_limit_tokens,
            });
        }
        let inner = self
            .model
            .new_cache(
                request.capacity_tokens(),
                request.max_cache_bytes(),
                &self.stream,
            )
            .map_err(|e| mlx_model_error(e, control, "allocate hybrid state"))?;
        control.check_cancelled()?;
        Ok(Box::new(HybridCache {
            inner,
            model: Rc::clone(&self.model),
            stream: Rc::clone(&self.stream),
            owner: self.instance.clone(),
            mode: request.mode().clone(),
        }))
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        PrefillOutput::new(tokens.len(), self.forward(tokens, cache, control)?)
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        if cache.len() == 0 {
            return Err(EngineError::invalid_request(
                "hybrid cache",
                "prefill before decode",
            ));
        }
        self.forward(&[token], cache, control)
    }
}
