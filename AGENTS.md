# Agent & Developer Quick Reference

Quick reference for coding agents working in this repository.

**For user-facing API and usage examples see `README.md`.**
**For format specification see `format/SPECIFICATION.md`.**

---

## 1. Build & Test Commands

### Scala (sbt)

```bash
# Compile
sbt compile

# All unit tests
sbt test

# Single test file
sbt "testOnly *SafetensorsDtypeSpec"

# Single test case (substring match on description)
sbt "testOnly *SafetensorsHeaderParserSpec -- -z 'parse a scalar'"

# Different Spark version
sbt -DsparkVersion=4.0.1 test

# Check formatting (CI fails if this fails)
sbt scalafmtCheck

# Apply formatting
sbt scalafmt

# Build fat JAR (for cluster/integration tests)
sbt assembly
# Output: target/scala-2.13/safetensors-spark-assembly-*.jar
```

### Python integration tests (uv + pytest)

```bash
# Install test dependencies
uv sync --project tests/pyspark_interop

# All integration tests (requires fat JAR)
SAFETENSORS_SPARK_JAR=target/scala-2.13/safetensors-spark-assembly-*.jar \
  uv run --project tests/pyspark_interop pytest tests/pyspark_interop/

# Single test file
uv run --project tests/pyspark_interop \
  pytest tests/pyspark_interop/test_python_to_spark.py

# Single test case
uv run --project tests/pyspark_interop \
  pytest tests/pyspark_interop/test_python_to_spark.py::test_schema_matches_tensor_struct
```

---

## 2. Package Layout

```
src/main/scala/io/github/semyonsinchenko/safetensors/
  core/           # SafetensorsDtype, TensorSchema, Header parsing/writing
  read/           # Spark read path: Scan, PartitionReader
  write/          # Spark write path: WriteBuilder, DataWriter
  expressions/    # SQL functions: arr_to_st(), st_to_array()
  manifest/       # DatasetManifest for MLflow lineage
  util/           # Errors helper for Spark 4 compatibility
  SafetensorsTable.scala        # Table with SupportsRead/SupportsWrite
  SafetensorsTableProvider.scala # DataSourceRegister ("safetensors")

src/test/scala/  # Unit tests: *Spec.scala files
python/safetensors_spark/  # Python utilities
tests/pyspark_interop/      # Integration tests
```

---

## 3. Scala Code Style

### Formatting
- Max line length: **100 columns**, indent: **2 spaces**
- Enforced by `.scalafmt.conf` (`sbt scalafmtCheck` / `sbt scalafmt`)
- Modifier order: `override`, `private/protected`, `implicit`, `final`, `sealed`, `abstract`, `lazy`
- `-Xfatal-warnings` is active; all warnings are errors.

### Import ordering (3 groups, blank line between)
1. Project-internal, 2. Spark/Hadoop/Jackson (provided deps), 3. Java/Scala stdlib

### Naming
| Construct | Convention | Example |
|-----------|------------|---------|
| Classes, objects, traits | PascalCase | `SafetensorsTableProvider` |
| Methods, val, var, parameters | camelCase | `headerSize`, `buildForBatch` |
| Module-level constants | UPPER_SNAKE | `DATA_FIELD`, `MIN_SHARD_SIZE_MB` |
| Type parameters | Single letter | `T`, `K`, `V` |

### null and Option
Never return `null`. Wrap Java APIs: `Option(opts.get("key")).map(...)`
Use `Option[T]` in signatures; prefer `getOrElse` / `fold` over `.get`.

### Collections
- Use `Seq[T]` (immutable) in public signatures
- Use `Map[K, V]` (immutable) for mappings
- Use `Array[T]` only for JVM interop (InternalRow, ByteBuffer, partitions)

### Error handling
| Situation | Pattern |
|-----------|---------|
| Plan validation | `throw Errors.analysisException(msg)` |
| Precondition | `require(cond, msg)` or `throw new IllegalArgumentException(msg)` |
| Expected failure | Return `Either[String, T]` |
| Catch exceptions | `case NonFatal(e) =>` |

**Never** call `new AnalysisException(msg)` directly — use `Errors.analysisException()`.

### Scaladoc
- Required on all public types (describe purpose, not implementation)
- Required on non-obvious public methods (use @param, @return, @throws)
- Private helpers: inline `//` comments sufficient
- Format: `/** ... */` (not `/* */`)

---

## 4. Python Code Style
```python
from __future__ import annotations  # always first
import json
from pathlib import Path
from typing import Optional
# third-party (defer heavy imports inside functions)
```
- All public functions need type hints (use `Optional[T]`, not `T | None`)
- NumPy-style docstrings; naming: `snake_case` (functions), `PascalCase` (classes)

---

## 5. Architectural Invariants

| Invariant | Rationale |
|-----------|-----------|
| BF16 hardcoded as valid dtype | Not in JSON schema regex; every acceptance site needs comment |
| Use `Errors.analysisException()` | Spark 4 removed single-String constructor |
| Write I/O via Hadoop FileSystem | Required for HDFS/S3/GCS/Blob compatibility |
| One .safetensors file = one InputPartition | Full header must be read first; no mid-file splits |
| Tensor bytes never heap-copied | Read: slice MappedByteBuffer; Write: ByteBuffer.wrap(bytes) |
| Schema validation in WriteBuilder.buildForBatch() | Errors at plan time, not task runtime |
| Index file is `_tensor_index.parquet` | `_metadata.parquet` reserved by Spark |

---

## 6. No Cursor/Copilot Rules

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` files found.
