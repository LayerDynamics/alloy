require "minitest/autorun"
require "open3"
require "yaml"

class CiPartitionTest < Minitest::Test
  def setup
    @jobs = YAML.load_file(File.expand_path("../../.github/workflows/ci.yaml", __dir__)).fetch("jobs")
  end

  def commands(job)
    @jobs.fetch(job).fetch("steps").filter_map { |step| step["run"] }
      .join("\n").gsub(/\\\s*\n\s*/, " ").gsub(/\s+/, " ")
  end

  def test_linux_shards_keep_required_checks_and_candidate_evidence
    assert_equal %w[linux_foundation linux_repro linux_engine linux_core], @jobs.fetch("linux").fetch("needs")
    assert_equal "linux-arm64", @jobs.fetch("linux").fetch("name")
    foundation = commands("linux_foundation")
    %w[pins_check.sh bench/corpus/validate.sh bindgen_check.sh coverage_inventory.sh].each do |gate|
      assert_includes foundation, gate
    end
    assert_includes commands("linux_repro"), "bash scripts/repro_check.sh --test"
    engine = commands("linux_engine")
    ["cargo test --locked -p tool-registry --all-targets", "cargo test --locked -p tool-registry --doc",
     "cargo test --locked -p mlx-ffi --all-targets -- --test-threads=1",
     "cargo test --locked -p inference-infra --all-targets", "cargo test --locked -p qwen3 --all-targets",
     "cargo test --locked -p qwen3 --doc", "cargo check --locked -p engine --all-targets",
     "cargo test --locked -p engine --all-targets", "cargo test --locked -p engine --doc",
     'RUSTDOCFLAGS="-D warnings" cargo doc --locked -p engine --no-deps'].each { |gate| assert_includes engine, gate }
    core = commands("linux_core")
    ["cargo clippy --locked --workspace --all-targets -- -D warnings", "cargo test --locked -p harness-core --all-targets",
     "cargo test --locked -p harness-core --doc", "cargo test --locked -p alloyd --all-targets",
     "cargo test --locked -p alloyd --doc", "bash scripts/builtin_acceptance.sh", "bash scripts/plugin_acceptance.sh",
     'ALLOY_W5_CANDIDATE="${{ github.sha }}"', 'ALLOY_W4_CANDIDATE="${{ github.sha }}"'].each do |gate|
      assert_includes core, gate
    end
    assert_equal %w[w5-linux w4-linux], @jobs.fetch("linux_core").fetch("steps")
      .filter_map { |step| step.dig("with", "name")&.split("-${{")&.first }
    @jobs.fetch("linux_core").fetch("steps").select { |step| step.dig("with", "name") }.each do |upload|
      assert_equal "error", upload.fetch("with").fetch("if-no-files-found")
    end
  end

  def test_mac_shards_keep_required_checks_and_aggregate_fails_closed
    assert_equal %w[mac_core mac_integration], @jobs.fetch("mac").fetch("needs")
    assert_equal "mac-metal", @jobs.fetch("mac").fetch("name")
    core = commands("mac_core")
    ["cargo test --locked -p tool-registry --all-targets", "cargo test --locked -p tool-registry --doc",
     "cargo test --locked -p mlx-ffi --no-fail-fast -- --test-threads=1", "-p inference-infra --all-targets",
     "-p qwen3 --all-targets", "-p qwen3 --doc", "cargo check --locked -p engine --all-targets",
     "cargo test --locked -p engine --all-targets", "cargo test --locked -p engine --doc",
     "cargo clippy --locked --workspace --all-targets -- -D warnings",
     "bash scripts/builtin_acceptance.sh", "bash scripts/plugin_acceptance.sh",
     "-p alloyd --all-targets", "-p alloyd --doc"].each { |gate| assert_includes core, gate }
    collector = File.read(File.expand_path("../plugin_acceptance.sh", __dir__))
    assert_includes collector, "gate all-targets cargo test --locked -p harness-core --all-targets"
    assert_includes collector, "gate doctests cargo test --locked -p harness-core --doc"
    integration = commands("mac_integration")
    %w[qwen3_end_to_end_stream.sh session_restart_real_model.sh large_model_acceptance.sh scripts/homebrew/ci.mjs].each do |gate|
      assert_includes integration, gate
    end
    assert_includes integration, "npm ci"
    %w[mac_core mac_integration].each do |shard|
      assert_equal "python3 scripts/ci/mac_locked_step.py {0}", @jobs.fetch(shard).dig("defaults", "run", "shell")
    end
    cleanup = @jobs.fetch("mac_integration").fetch("steps").find { |step| step["name"] == "Clean this run's owned Homebrew resources" }
    assert_equal "python3 scripts/ci/mac_locked_step.py --cleanup {0}", cleanup.fetch("shell")
    assert_equal "exec node scripts/homebrew/ci.mjs --cleanup-owned", cleanup.fetch("run")
    @jobs.fetch("mac_core").fetch("steps").select { |step| step.dig("with", "name") }.each do |upload|
      assert_equal "error", upload.fetch("with").fetch("if-no-files-found")
    end
    %w[linux mac].each do |aggregate|
      job = @jobs.fetch(aggregate)
      assert_includes job.fetch("if"), "always()"
      job.fetch("needs").each { |needed| assert_includes job.fetch("steps").first.fetch("env").values.join(" "), "needs.#{needed}.result" }
      assert_equal job.fetch("needs").length, commands(aggregate).scan(/= success/).length
      env_names = job.fetch("steps").first.fetch("env").keys
      env_names.product(%w[failure cancelled skipped]).each do |changed, result|
        env = env_names.to_h { |name| [name, name == changed ? result : "success"] }
        _, status = Open3.capture2e(env, "bash", "-c", commands(aggregate))
        refute status.success?, "#{aggregate} accepted #{changed}=#{result}"
      end
      _, status = Open3.capture2e(env_names.to_h { |name| [name, "success"] }, "bash", "-c", commands(aggregate))
      assert status.success?, "#{aggregate} rejected all successful shards"
    end
  end
end
