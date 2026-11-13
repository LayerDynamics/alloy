require "minitest/autorun"
require "open3"
require "yaml"
require "tmpdir"
require "fileutils"

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
     "cargo test --locked -p qwen3 --doc", "cargo test --locked -p llama3 --all-targets", "cargo test --locked -p llama3 --doc", "cargo check --locked -p engine --all-targets",
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
    ["python3 scripts/approval_storage_evidence.py", '--repo "$PWD" --expected-commit "$CI_CANDIDATE" --output "$evidence/results"',
     "cargo test --locked --workspace --exclude mlx-ffi --exclude harness-core --all-targets",
     "bash scripts/builtin_acceptance.sh", "bash scripts/plugin_acceptance.sh"].each { |gate| assert_includes core, gate }
    assert_equal 1, core.scan("python3 scripts/approval_storage_evidence.py").length
    refute_includes core, "--capture-only"
    assert_includes core, 'evidence="$RUNNER_TEMP/approval-storage-evidence-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-macos"'
    approval = File.read(File.expand_path("../approval_storage_evidence.py", __dir__))
    [ '["cargo", "fmt", "--all", "--", "--check"]',
      '["cargo", "check", "--locked", "--workspace", "--all-targets"]',
      '["cargo", "clippy", "--locked", "--workspace", "--all-targets", "--", "-D", "warnings"]',
      '["cargo", "test", "--locked", "--workspace", "--all-targets", "--", "--test-threads=1"]',
      '["cargo", "test", "--locked", "--workspace", "--doc"]',
      '["cargo", "test", "--locked", "-p", "harness-core", "--all-targets"]',
      '["cargo", "doc", "--locked", "--workspace", "--no-deps"]',
      'env["RUSTDOCFLAGS"] = "-D warnings"',
      'destructive_effect_requires_bound_approval', 'oversized_tool_result_artifact_roundtrip',
      'approval_storage_boundary_duplicate_keys_fail_in_atomic_and_streaming_modes',
      'approval_storage_boundary_five_frozen_builtin_defaults_preserve_prompt_resolution',
      'freeze_manifest_matches_normative_record', 'frozen_vectors_are_reproduced_by_rust',
    ].each { |gate| assert_includes approval, gate }
    mac_candidate = "${{ github.event.pull_request.head.sha || github.sha }}"
    %w[mac_core mac_integration].each do |shard|
      job = @jobs.fetch(shard)
      assert_equal mac_candidate, job.fetch("env").fetch("CI_CANDIDATE")
      checkout = job.fetch("steps").find { |step| step["uses"] == "actions/checkout@v5" }.fetch("with")
      assert_equal mac_candidate, checkout.fetch("ref")
      assert_equal 0, checkout.fetch("fetch-depth")
      assert_equal "recursive", checkout.fetch("submodules")
      assert_equal false, checkout.fetch("clean")
    end
    @jobs.fetch("mac_core").fetch("steps").select { |step| step.fetch("env", {}).keys.any? { |key| key.match?(/ALLOY_W[45]_CANDIDATE/) } }.each do |step|
      candidates = step.fetch("env").select { |key, _| key.match?(/ALLOY_W[45]_CANDIDATE/) }.values
      assert_equal [mac_candidate], candidates
      assert_includes step.fetch("run"), 'test "$(git rev-parse HEAD)" = "$CI_CANDIDATE"'
      assert_includes step.fetch("run"), '"$CI_CANDIDATE" "$CI_PR_HEAD"'
      refute_includes step.fetch("run"), '"$GITHUB_SHA"'
    end
    old_jobs = YAML.load_file(File.expand_path("../../.github/workflows/approval-storage-evidence.yaml", __dir__)).fetch("jobs")
    assert_equal "approval-storage-macos-standalone", old_jobs.fetch("mac").fetch("name")
    assert_equal "github.event_name != 'pull_request'", old_jobs.fetch("mac").fetch("if").strip
    assert_equal 1, (@jobs.values + old_jobs.values).count { |job| job["name"] == "approval-storage-macos-arm64" }
    assert_equal ["mac_core"], @jobs.fetch("approval_storage_mac").fetch("needs")
    assert_equal "${{ always() }}", @jobs.fetch("approval_storage_mac").fetch("if")
    approval_upload = @jobs.fetch("mac_core").fetch("steps").find { |step| step.dig("with", "name")&.start_with?("approval-storage-macos-") }
    assert_equal "${{ always() }}", approval_upload.fetch("if")
    assert_includes approval_upload.fetch("with").fetch("path"), "${{ runner.temp }}/approval-storage-evidence-"
    assert_equal 30, approval_upload.fetch("with").fetch("retention-days")
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
    %w[linux mac approval_storage_mac].each do |aggregate|
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
  def test_mac_checkout_removes_stale_source_but_preserves_only_root_cargo_cache
    %w[mac_core mac_integration].each do |shard|
      steps = @jobs.fetch(shard).fetch("steps")
      cleanup = steps.find { |step| step["name"] == "Clean source while retaining the Cargo build cache" }
      assert_equal "clean_source", cleanup.fetch("id")
      steps.select { |step| step.dig("with", "path")&.start_with?("target/") }.each do |upload|
        assert_equal "${{ always() && steps.clean_source.outcome == 'success' }}", upload.fetch("if")
      end
      assert_operator steps.index(cleanup), :>, steps.index { |step| step["uses"] == "actions/checkout@v5" }
      assert_operator steps.index(cleanup), :<, steps.index { |step| step["name"] == "Node runtime" }
      Dir.mktmpdir("alloy-ci-checkout-") do |root|
        run = lambda do |*argv|
          output, status = Open3.capture2e(*argv, chdir: root)
          assert status.success?, output
          output.strip
        end
        run.call("git", "init", "-q")
        File.write(File.join(root, "tracked"), "candidate\n")
        File.write(File.join(root, ".gitignore"), "target/\nignored-source\n")
        run.call("git", "add", "tracked", ".gitignore")
        run.call("git", "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        candidate = run.call("git", "rev-parse", "HEAD")
        File.write(File.join(root, "tracked"), "stale edit\n")
        File.write(File.join(root, "ignored-source"), "must remove")
        File.write(File.join(root, "untracked-source"), "must remove")
        %w[debug release].each do |profile|
          FileUtils.mkdir_p(File.join(root, "target", profile))
          File.write(File.join(root, "target", profile, "cache"), "compiled output")
        end
        FileUtils.mkdir_p(File.join(root, "stale", "target"))
        File.write(File.join(root, "stale", "target", "foreign"), "must remove")
        %w[w4-acceptance w5-acceptance phase-1-acceptance].each do |name|
          FileUtils.mkdir_p(File.join(root, "target", name))
          File.write(File.join(root, "target", name, "stale-proof"), "old candidate")
        end
        output, status = Open3.capture2e({"CI_CANDIDATE" => candidate}, "bash", "-c", cleanup.fetch("run"), chdir: root)
        assert status.success?, output
        assert_equal "candidate\n", File.read(File.join(root, "tracked"))
        %w[debug release].each do |profile|
          assert_equal "compiled output", File.read(File.join(root, "target", profile, "cache"))
        end
        %w[ignored-source untracked-source stale].each { |name| refute File.exist?(File.join(root, name)), name }
        %w[w4-acceptance w5-acceptance phase-1-acceptance].each do |name|
          refute File.exist?(File.join(root, "target", name)), name
        end
        assert_equal "", run.call("git", "status", "--porcelain=v1", "--untracked-files=all")
      end
    end
  end

end
