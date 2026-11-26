use tool_registry::execution::{BoundedOutput, ExecutionContext, ExecutionError, OutputSink};

pub struct DeadlineOnChildOutput<'a> {
    pub context: &'a ExecutionContext,
    pub output: &'a mut BoundedOutput,
}

impl OutputSink for DeadlineOnChildOutput<'_> {
    fn write_chunk(&mut self, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.output.write_chunk(bytes)?;
        // The fixture's test harness writes stdout only after native launch.
        self.context.advance_to_deadline_for_test();
        Ok(())
    }

    fn abort(&mut self) {
        self.output.abort();
    }
}
