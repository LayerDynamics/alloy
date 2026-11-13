// Physical extraction: intentionally shares qwen.rs scope.

impl ModelFamily for Qwen3Backend {
    fn registration(&self) -> &FamilyRegistration {
        &self.registration
    }

    fn load(&self, request: ModelLoadRequest<'_>) -> Result<Box<dyn LoadedModel>, EngineError> {
        let control = request.control();
        control.check_cancelled()?;
        let data = self.data(request.artifact())?;
        let decoded_bound = data.decoded_bound;
        // Headroom covers group-size variance (planning assumes 64; some
        // fixtures/checkpoints use smaller groups with larger scale tables).
        let weight_charge_limit = sum(&[decoded_bound, self.limits.loader_headroom_bytes])?;
        let required_peak = loading_peak_bound(data, &self.limits)?;
        if request.limits().max_source_bytes() < data.source_bytes
            || request.limits().max_resident_bytes() < data.resident_bound
            || request.limits().max_load_peak_bytes() < required_peak
        {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "MLX Qwen loading overlap",
                required_bytes: required_peak,
                available_bytes: request.limits().max_load_peak_bytes(),
                reason: MemoryLimitReason::BufferLimit,
            });
        }

        let cpu_device = Device::new(DeviceType::Cpu, 0)
            .map_err(|e| execution_error("create MLX CPU device", e.to_string()))?;
        if !cpu_device
            .is_available()
            .map_err(|e| execution_error("query MLX CPU", e.to_string()))?
        {
            return Err(EngineError::unsupported(
                "MLX CPU",
                "requested device is unavailable",
            ));
        }
        let loading_stream = Stream::new(&cpu_device)
            .map_err(|e| execution_error("create MLX loading stream", e.to_string()))?;
        let execution_stream = match self.execution {
            Qwen3Execution::MlxCpu => Rc::new(loading_stream),
            Qwen3Execution::MlxMetal => {
                if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    return Err(EngineError::unsupported(
                        "MLX Metal",
                        "requires an Apple-Silicon macOS worker",
                    ));
                }
                let gpu = Device::new(DeviceType::Gpu, 0)
                    .map_err(|e| execution_error("create MLX Metal device", e.to_string()))?;
                if !gpu
                    .is_available()
                    .map_err(|e| execution_error("query MLX Metal", e.to_string()))?
                {
                    return Err(EngineError::unsupported(
                        "MLX Metal",
                        "requested device is unavailable; no CPU fallback",
                    ));
                }
                // Array construction copies decoded source buffers; the
                // model subsequently evaluates its arrays on this worker.
                // The unused CPU setup stream has no submitted operations.
                drop(loading_stream);
                Rc::new(
                    Stream::new(&gpu)
                        .map_err(|e| execution_error("create MLX Metal stream", e.to_string()))?,
                )
            }
        };
        control.check_cancelled()?;
        let cancelled = || control.is_cancelled();
        let verification = CheckpointVerification {
            sha256: &data.checksums,
            cancelled: &cancelled,
        };
        let result = load_weights_verified(
            &data.config,
            &data.specification.directory,
            LoadOptions {
                // Enforce the source limit used by the workspace plan even
                // when a caller offers a larger grant. Mutated files must
                // not expand metadata beyond that bound before hash failure.
                max_file_bytes: as_usize(data.source_bytes)?,
                max_weight_bytes: as_usize(weight_charge_limit)?,
            },
            &verification,
        );
        control.check_cancelled()?;
        let weights = result.map_err(|e| model_error(e, "load MLX Qwen weights"))?;
        let packed_bits = weights.embed_tokens.bits();
        if !data.config.has_qk_norm() {
            validate_llama_weight_layout(&weights)?;
        }
        let decoded = weight_memory(&weights)?;
        if decoded.allocated_bytes() > weight_charge_limit {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "decoded MLX Qwen weight buffers",
                required_bytes: decoded.allocated_bytes(),
                available_bytes: weight_charge_limit,
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        let model = MlxQwen3Model::from_weights(
            data.config.clone(),
            weights,
            request.limits().max_resident_bytes(),
            execution_stream.as_ref(),
        )
        .map_err(|e| mlx_model_error(e, control, "construct MLX Qwen model"))?;
        control.check_cancelled()?;
        let payload = model.weight_payload_bytes();
        let measured = model
            .weight_buffer_bytes()
            .map_err(|e| mlx_model_error(e, control, "measure MLX Qwen weights"))?;
        if measured > request.limits().max_resident_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "measured MLX Qwen weight buffers",
                required_bytes: measured,
                available_bytes: request.limits().max_resident_bytes(),
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        let mut capabilities = self.registration.capabilities().clone();
        if !data.config.has_qk_norm() {
            capabilities.weight_representations = vec![match packed_bits {
                Some(8) => WeightRepresentation::Affine8 { group_size: NonZeroUsize::new(64).unwrap() },
                Some(4) => WeightRepresentation::Affine4 { group_size: NonZeroUsize::new(64).unwrap() },
                _ => WeightRepresentation::F32,
            }];
        }
        capabilities.context_limit_tokens = capabilities
            .context_limit_tokens
            .min(data.config.max_context);
        Ok(Box::new(Qwen3Loaded {
            model,
            stream: execution_stream,
            artifact: request.artifact().id().clone(),
            instance: ModelInstanceId::new(),
            registration: self.registration.clone(),
            capabilities,
            memory: BufferMemoryUsage::new(measured, payload)?,
            vocabulary: data.config.vocab_size,
            max_prefill_tokens: self.limits.prefill_chunk_tokens.get(),
            logits_payload_bytes: product(logits_payload_bound(&data.config)?, if data.config.has_qk_norm() { 1 } else { as_u64(self.limits.prefill_chunk_tokens.get())? })?,
        }))
    }
}

struct Qwen3Loaded {
    model: MlxQwen3Model,
    stream: Rc<Stream>,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    // Tracks float32 tensor payload, not allocator cache or process RSS.
    memory: BufferMemoryUsage,
    vocabulary: usize,
    max_prefill_tokens: usize,
    logits_payload_bytes: u64,
}

impl LoadedModel for Qwen3Loaded {
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

impl Drop for Qwen3Loaded {
    fn drop(&mut self) {
        // Normal model operations already synchronize on every return.
        // This is best-effort teardown, not native device-failure recovery.
        let _ = self.stream.synchronize();
    }
}

impl Qwen3Loaded {
    fn owned_cache<'a>(
        &self,
        cache: &'a mut dyn FamilyCache,
    ) -> Result<&'a mut Qwen3NativeCache, EngineError> {
        if cache.owner() != &self.instance || cache.mode() != &family_cache_mode(self.model.config().family)? {
            return Err(EngineError::invalid_request(
                "cache",
                "cache belongs to another model or representation",
            ));
        }
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Qwen3NativeCache>()
            .ok_or_else(|| {
                EngineError::invalid_request("cache", "cache has a different family type")
            })?;
        if !cache.inner.owner().same_instance(self.model.instance_id())
            || !cache.mlx_owner.same_instance(self.model.instance_id())
        {
            return Err(EngineError::invalid_request(
                "cache",
                "native MLX cache belongs to another model load",
            ));
        }
        Ok(cache)
    }
}

impl AutoregressiveModel for Qwen3Loaded {
    fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        control.check_cancelled()?;
        if request.mode() != &family_cache_mode(self.model.config().family)? {
            return Err(EngineError::unsupported(
                "cache mode",
                "only full MLX float32 KV is implemented",
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
        let required = cache_bound(self.model.config(), request.capacity_tokens())?;
        if required > request.max_cache_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "MLX Qwen KV update overlap",
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
                self.stream.as_ref(),
            )
            .map_err(|e| mlx_model_error(e, control, "allocate MLX Qwen KV"))?;
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
        let cache = self.owned_cache(cache)?;
        control.check_cancelled()?;
        if tokens.is_empty() || tokens.len() > self.max_prefill_tokens {
            return Err(EngineError::invalid_request(
                "tokens",
                "prefill must be nonempty and fit the planned chunk bound",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("prefill position overflow"))?;
        if end > cache.capacity {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.capacity,
            });
        }
        let result = self
            .model
            .prefill_cancellable(
                tokens,
                &mut cache.inner,
                self.logits_payload_bytes,
                self.stream.as_ref(),
                || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "MLX Qwen prefill"))
            .and_then(|output| TokenLogits::new(output.into_values()))
            .and_then(|logits| PrefillOutput::new(tokens.len(), logits));
        if result.is_err() {
            let _ = cache.inner.reset(&cache.mlx_owner);
        }
        result
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        let cache = self.owned_cache(cache)?;
        control.check_cancelled()?;
        if cache.inner.is_empty() {
            return Err(EngineError::invalid_request(
                "cache",
                "prefill before decoding a continuation",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(1)
            .ok_or_else(|| internal("decode position overflow"))?;
        if end > cache.capacity {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.capacity,
            });
        }
        // Use the cancellable one-token path, not the noncancellable
        // convenience decode method. It has identical model semantics.
        let result = self
            .model
            .prefill_cancellable(
                &[token],
                &mut cache.inner,
                self.logits_payload_bytes,
                self.stream.as_ref(),
                || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "MLX Qwen decode"))
            .and_then(|output| TokenLogits::new(output.into_values()));
        if result.is_err() {
            let _ = cache.inner.reset(&cache.mlx_owner);
        }
        result
    }
}


fn validate_llama_weight_layout(weights: &qwen3::fixture::Qwen3Weights) -> Result<(), EngineError> {
    let packed_bits = weights.embed_tokens.bits();
    let matrices = std::iter::once(&weights.embed_tokens)
        .chain(weights.lm_head.iter())
        .chain(weights.layers.iter().flat_map(|l| [&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj, &l.gate_proj, &l.up_proj, &l.down_proj]));
    for matrix in matrices {
        if matrix.bits() != packed_bits || matrix.affine4_group_size().is_some_and(|group| group != 64) {
            return Err(EngineError::unsupported("Llama weight layout", "requires uniform dense weights or uniform affine group-size 64 matrices"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod llama_layout_tests {
    use super::*;
    use qwen3::fixture::{MatrixWeight, generate_fixture};
    #[test]
    fn llama_rejects_mixed_and_unqualified_groups_before_publication() {
        let mut weights = generate_fixture(&Qwen3Config::tiny_fixture());
        assert!(validate_llama_weight_layout(&weights).is_ok());
        weights.layers[0].q_proj = MatrixWeight::Affine8 { packed: vec![], scales: vec![], biases: vec![], rows: 64, cols: 64, group_size: 64 };
        assert!(validate_llama_weight_layout(&weights).is_err());
        weights.embed_tokens = MatrixWeight::Affine4 { packed: vec![], scales: vec![], biases: vec![], rows: 64, cols: 64, group_size: 32 };
        assert!(validate_llama_weight_layout(&weights).is_err());
    }
}
