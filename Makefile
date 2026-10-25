# One command gates everything: `make check`.
# Honors CARGO_TARGET_DIR / CARGO_BUILD_JOBS from the environment.
CARGO ?= cargo
COVERAGE_MIN ?= 80

.PHONY: check fmt fmt-check lint doc doctest test cov demo bench bench-fixture bench-sift fetch-data loc clean

check: fmt-check lint doc doctest cov ## format + clippy (zero warnings) + rustdoc + doc examples + tests with coverage gate

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

lint:
	$(CARGO) clippy --all-targets --locked -- -D warnings

doc:
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --locked

doctest:
	$(CARGO) test --doc --locked

test:
	$(CARGO) test --locked

# Runs the whole test suite (unit, integration, property, CLI) under
# instrumentation and fails below COVERAGE_MIN percent line coverage.
cov:
	$(CARGO) llvm-cov --locked --summary-only --fail-under-lines $(COVERAGE_MIN)

demo:
	$(CARGO) run --release --locked -- demo --fixtures fixtures --workdir demo-out

fetch-data:
	./scripts/fetch-sift.sh data

bench: bench-fixture bench-sift

bench-fixture:
	$(CARGO) run --release --locked -- bench --dataset fixture --data fixtures \
		--m 16 --ef-construction 200 --k 10 --out bench/results/fixture

bench-sift: fetch-data
	$(CARGO) run --release --locked -- bench --dataset siftsmall --data data/siftsmall \
		--m 16 --ef-construction 200 --k 10 --out bench/results/siftsmall

loc:
	@echo "non-test source lines (src/, excluding #[cfg(test)] modules):"
	@for f in $$(find src -name '*.rs' | sort); do \
		awk '/^#\[cfg\(test\)\]/{exit} {n++} END{printf "%6d %s\n", n, FILENAME}' $$f; \
	done | tee /dev/stderr | awk '{s+=$$1} END{print "total", s}'

clean:
	rm -rf demo-out bench/tmp
