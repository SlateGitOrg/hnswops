//! End-to-end tests of the `hnswops` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_hnswops"));
    c.current_dir(env!("CARGO_MANIFEST_DIR"));
    c.env("HNSWOPS_LOG", "error");
    c
}

fn run(args: &[&str]) -> Output {
    bin().args(args).output().expect("spawn hnswops")
}

fn ok(args: &[&str]) -> String {
    let out = run(args);
    assert!(
        out.status.success(),
        "hnswops {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn json(args: &[&str]) -> serde_json::Value {
    serde_json::from_str(&ok(args)).expect("valid json")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn build_eval_search_snapshot_lifecycle() {
    let tmp = tempfile::tempdir().unwrap();
    let store: PathBuf = tmp.path().join("store");
    let st = s(&store);
    assert!(ok(&["verify-data"]).contains("10000 base x 32d"));
    let out = ok(&["build", "--store", st, "--m", "12", "--ef-construction", "100", "--label", "first"]);
    assert!(out.contains("committed snapshot v1"), "{out}");

    let eval = json(&["eval", "--store", st, "--k", "10", "--ef", "64"]);
    assert_eq!(eval["mapped_vectors"], 10_000);
    assert!(eval["recall"].as_f64().unwrap() >= 0.99, "{eval}");

    let hit = json(&["search", "--store", st, "--query", "3", "--k", "5", "--filter", "tags_any=0x100"]);
    assert_eq!(hit["strategy"], "brute_force");
    let keys: Vec<u64> = hit["hits"].as_array().unwrap().iter().map(|h| h["key"].as_u64().unwrap()).collect();
    assert_eq!(keys.len(), 5);
    assert!(keys.iter().all(|k| k % 50 == 0), "flag filter violated: {keys:?}");
    assert_eq!(hit["exact_keys"].as_array().unwrap().len(), 5);
    let plain = json(&["search", "--store", st, "--query", "0", "--ef", "32"]);
    assert_eq!(plain["strategy"], "graph");

    ok(&["build", "--store", st, "--m", "8", "--ef-construction", "40", "--label", "second"]);
    let list = ok(&["snapshot", "--store", st, "list"]);
    assert!(list.contains("* v000002") && list.contains("label=first"), "{list}");
    assert!(ok(&["snapshot", "--store", st, "verify", "2"]).contains("v000002 ok"));
    assert!(ok(&["snapshot", "--store", st, "rollback", "1"]).contains("CURRENT -> v000001"));
    let health = json(&["health", "--store", st]);
    assert_eq!(health["version"], 1);
    // A fresh build may orphan a handful of nodes (see `repair_connectivity`).
    assert!(health["health"]["unreachable_l0"].as_u64().unwrap() < 20, "{health}");
    assert!(ok(&["snapshot", "--store", st, "prune", "--keep", "0"]).contains("[2]"));
}

#[test]
fn errors_exit_nonzero_with_message() {
    let tmp = tempfile::tempdir().unwrap();
    let st = s(tmp.path());
    for args in [
        vec!["eval", "--store", st],
        vec!["snapshot", "--store", st, "verify", "7"],
        vec!["verify-data", "--data", st],
        vec!["verify-data", "--dataset", "siftsmall", "--data", "fixtures"],
        vec!["build", "--store", st, "--m", "1"],
    ] {
        let out = run(&args);
        assert!(!out.status.success(), "{args:?} should fail");
        assert!(String::from_utf8_lossy(&out.stderr).starts_with("error: "), "{args:?}");
    }
    ok(&["build", "--store", st, "--m", "4", "--ef-construction", "8"]);
    let out = run(&["search", "--store", st, "--query", "999999"]);
    assert!(!out.status.success());
    let out = run(&["search", "--store", st, "--filter", "colour=red"]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("bad filter clause"));
}

#[test]
fn gen_fixture_and_bench_write_outputs() {
    let tmp = tempfile::tempdir().unwrap();
    let fx = tmp.path().join("fx");
    let out = ok(&["gen-fixture", "--out", s(&fx), "--n", "500", "--queries", "20", "--dim", "8", "--k", "10"]);
    assert_eq!(out.lines().count(), 3);
    // The generated files are not the pinned fixture, so the pinned spec rejects them…
    assert!(!run(&["verify-data", "--data", s(&fx)]).status.success());
    // …and the committed fixture benchmarks fine.
    let prefix = tmp.path().join("res").join("fixture");
    let md = ok(&[
        "bench", "--m", "8", "--ef-construction", "32", "--efs", "8,32", "--min-ms", "1", "--out", s(&prefix),
    ]);
    assert!(md.contains("| 32 |"), "{md}");
    for ext in ["json", "md", "svg"] {
        assert!(prefix.with_extension(ext).exists(), "{ext}");
    }
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(prefix.with_extension("json")).unwrap()).unwrap();
    assert_eq!(report["sweep"].as_array().unwrap().len(), 2);
}

#[test]
fn demo_runs_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let work = tmp.path().join("demo");
    let out = ok(&["demo", "--workdir", s(&work)]);
    for needle in [
        "== 8. zero-downtime re-embedding migration",
        "results identical to in-memory index: true",
        "deleted keys returned=0",
        "passed=true",
        "(0 failed)",
        "demo complete",
    ] {
        assert!(out.contains(needle), "missing {needle:?} in demo output:\n{out}");
    }
    assert!(work.join("index-store").join("CURRENT").exists());
    assert!(work.join("docs-store").join("versions").join("v000002").exists());
}
