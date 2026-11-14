// Native Phi execution shares engine policy but never the transformer model.
use inference_infra::decoder::phi::{PhiConfig, PhiRopeRegime, model::MlxPhiModel};

impl Qwen3Backend {
    fn load_phi(
        &self,
        request: ModelLoadRequest<'_>,
        data: &ArtifactData,
        config: &PhiConfig,
    ) -> Result<Box<dyn LoadedModel>, EngineError> {
        let control = request.control();
        control.check_cancelled()?;
        let peak = loading_peak_bound(data, &self.limits)?;
        if request.limits().max_source_bytes() < data.source_bytes
            || request.limits().max_resident_bytes() < data.resident_bound
            || request.limits().max_load_peak_bytes() < peak
        {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "native Phi load overlap",
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
        .map_err(|e| execution_error("create Phi device", e.to_string()))?;
        if !device
            .is_available()
            .map_err(|e| execution_error("query Phi device", e.to_string()))?
        {
            return Err(EngineError::unsupported(
                "Phi execution device",
                "requested device is unavailable; no fallback",
            ));
        }
        let stream = Rc::new(
            Stream::new(&device)
                .map_err(|e| execution_error("create Phi stream", e.to_string()))?,
        );
        let cancelled = || control.is_cancelled();
        let (loaded_config, weights) = inference_infra::decoder::phi::weights::load_verified(
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
        .map_err(|e| model_error(e, "load verified Phi weights"))?;
        control.check_cancelled()?;
        let model = MlxPhiModel::from_weights(
            loaded_config,
            weights,
            PhiRopeRegime::Long,
            request.limits().max_resident_bytes(),
            &stream,
        )
        .map_err(|e| mlx_model_error(e, control, "construct native Phi model"))?;
        control.check_cancelled()?;
        let memory = BufferMemoryUsage::new(
            model
                .weight_buffer_bytes()
                .map_err(|e| mlx_model_error(e, control, "measure Phi weights"))?,
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
                _ => return Err(internal("unvalidated Phi quantization")),
            }];
        Ok(Box::new(PhiLoaded {
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

struct PhiLoaded {
    model: MlxPhiModel,
    stream: Rc<Stream>,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    memory: BufferMemoryUsage,
    max_prefill_tokens: usize,
}
impl LoadedModel for PhiLoaded {
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
impl Drop for PhiLoaded {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl PhiLoaded {
    fn owned_cache<'a>(
        &self,
        cache: &'a mut dyn FamilyCache,
    ) -> Result<&'a mut Qwen3NativeCache, EngineError> {
        if cache.owner() != &self.instance
            || cache.mode() != &family_cache_mode(BackendFamily::Phi)?
        {
            return Err(EngineError::invalid_request(
                "cache",
                "foreign Phi model or position regime",
            ));
        }
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen3NativeCache>()
            .ok_or_else(|| EngineError::invalid_request("cache", "foreign Phi cache type"))?;
        if !cache.inner.owner().same_instance(self.model.instance_id())
            || !cache.mlx_owner.same_instance(self.model.instance_id())
        {
            return Err(EngineError::invalid_request(
                "cache",
                "foreign Phi native owner",
            ));
        }
        Ok(cache)
    }
    fn step(
        &self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        control.check_cancelled()?;
        let cache = self.owned_cache(cache)?;
        if tokens.is_empty() || tokens.len() > self.max_prefill_tokens {
            return Err(EngineError::invalid_request(
                "tokens",
                "Phi prefill exceeds admitted chunk",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("Phi position overflow"))?;
        if end > cache.capacity {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.capacity,
            });
        }
        let plan = self
            .model
            .config()
            .step_payload_plan(tokens.len(), end)
            .map_err(|e| mlx_model_error(e, control, "plan Phi step"))?;
        let result = self
            .model
            .forward_cached(
                tokens,
                &mut cache.inner,
                true,
                plan.output_bytes,
                plan.work_bytes,
                &self.stream,
                &mut || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "execute Phi step"))
            .and_then(|logits| TokenLogits::new(logits.into_values()));
        if result.is_err() {
            let _ = cache.inner.reset(&cache.mlx_owner);
        }
        result
    }
}
impl AutoregressiveModel for PhiLoaded {
    fn vocabulary_size(&self) -> usize {
        self.model.config().vocabulary
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        control.check_cancelled()?;
        if request.mode() != &family_cache_mode(BackendFamily::Phi)? {
            return Err(EngineError::unsupported(
                "Phi cache",
                "requires full KV with fixed long-factor position regime",
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
        let required = phi_cache_bound(self.model.config(), request.capacity_tokens())?;
        if required > request.max_cache_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "Phi KV overlap",
                required_bytes: required,
                available_bytes: request.max_cache_bytes(),
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        let inner = self
            .model
            .new_cache(
                request.capacity_tokens(),
                request.max_cache_bytes(),
                &self.stream,
            )
            .map_err(|e| mlx_model_error(e, control, "allocate Phi KV"))?;
        control.check_cancelled()?;
        Ok(Box::new(Qwen3NativeCache {
            inner,
            owner: self.instance.clone(),
            mlx_owner: self.model.instance_id().clone(),
            mode: request.mode().clone(),
            capacity: request.capacity_tokens(),
            stream: Rc::clone(&self.stream),
        }))
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        PrefillOutput::new(tokens.len(), self.step(tokens, cache, control)?)
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        control.check_cancelled()?;
        if self.owned_cache(cache)?.inner.is_empty() {
            return Err(EngineError::invalid_request(
                "cache",
                "prefill before Phi decode",
            ));
        }
        self.step(&[token], cache, control)
    }
}
