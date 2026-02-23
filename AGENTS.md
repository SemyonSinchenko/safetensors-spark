# Agent & Developer Quick Reference

Quick reference for coding agents and developers working in this repository.

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

# Different Spark version (Scala version inferred automatically)
sbt -DsparkVersion=4.0.1 test

CI will fail if this fails)
sbt scalafmt# Check formatting (Check

# Apply formatting
sbt scalafmt

# Build fat JAR tests
sbt assembly
# Output: target/scala for cluster / integration-2.13/safetensors-spark-assembly-*.jar
```

### Python integration tests (uv + pytest)

The integration tests require the fat JAR. Build it with `sbt assembly` first,
or point `SAFETENSORS_SPARK_JAR` at an existing JAR.

```bash
# Install test dependencies
uv sync --project tests/pyspark_interop

# All integration tests
SAFETENSORS_SPARK_JAR=target/scala-2.13/safetensors-spark-assembly-*.jar \
  uv run --project tests/pyspark_interop pytest tests/pyspark_interop/

# Single test file
uv run --project tests/pyspark_interop \
  pytest tests/pyspark_interop/test_python_to_spark.py

# Single test case
uv run --project tests/pyspark_interop \
  pytest tests/pyspark_interop/test_python_to_spark.py::test_schema_matches_tensor_struct

# Verbose output
uv run --project tests/pyspark_interop pytest tests/pyspark_interop/ -v --tb=short
```

---

## 2. Package Layout

```
src/main/scala/io/github/semyonsinchenko/safetensors/
  core/
    SafetensorsDtype.scala          sealed enum of all valid dtypes (incl. BF16 special case)
    TensorSchema.scala              canonical Tensor Struct definition and helpers
    SafetensorsHeader.scala         SafetensorsHeader, TensorInfo, DataOffsets case classes
    SafetensorsHeaderParser.scala   parse binary header from ByteBuffer → SafetensorsHeader
    SafetensorsHeaderWriter.scala   build binary header from TensorDescriptor list → ByteBuffer
  read/
    SafetensorsScanBuilder.scala    ScanBuilder + pushdown + column pruning
    SafetensorsScan.scala           Scan + Batch; plans InputPartitions
    SafetensorsInputPartition.scala one file path = one partition
    SafetensorsPartitionReaderFactory.scala
    SafetensorsPartitionReader.scala mmap read; one row per file
  write/
    WriteOptions.scala              parse + validate all write options eagerly
    SafetensorsWriteBuilder.scala   schema validation; produces SafetensorsBatchWrite
    SafetensorsBatchWrite.scala     driver-side commit; writes manifest + index
    SafetensorsDataWriterFactory.scala
    SafetensorsDataWriter.scala     executor-side tensor encoding + shard rolling
  expressions/
    ArrToStExpression.scala         arr_to_st() Catalyst expression
    StToArrayExpression.scala       st_to_array() Catalyst expression
  manifest/
    DatasetManifest.scala           DatasetManifest + ShardInfo case classes
  mlflow/
    SafetensorsDatasetSource.scala  JVM-side MLflow lineage utility
  util/
    Errors.scala                    Errors.analysisException() — Spark 4 compat helper
  SafetensorsTable.scala            Table + SupportsRead + SupportsWrite
  SafetensorsTableProvider.scala    TableProvider + DataSourceRegister ("safetensors")
  SafetensorsExtensions.scala       SparkSessionExtensions — registers SQL functions

src/main/resources/META-INF/services/
  org.apache.spark.sql.sources.DataSourceRegister   registers short name "safetensors"

src/test/scala/io/github/semyonsinchenko/safetensors/
  SafetensorsDtypeSpec.scala
  SafetensorsHeaderParserSpec.scala
  TensorSchemaSpec.scala

python/safetensors_spark/
  __init__.py
  mlflow.py                         log_dataset() — pure Python, no Spark dependency

tests/pyspark_interop/
  conftest.py                       session-scoped SparkSession + JAR fixture
  test_python_to_spark.py
  test_spark_to_python.py
  test_mlflow.py
```

---

## 3. Scala Code Style

### Formatting

Enforced by `.scalafmt.conf` (`sbt scalafmt` / `sbt scalafmtCheck`):
- Max line length: **100 columns**
- Indent: **2 spaces**
- Rewrites: `RedundantBraces`, `SortModifiers` (modifier order: `override`,
  `private/protected`, `implicit`, `final`, `sealed`, `abstract`, `lazy`)
- Align tokens: `->`, `<-`, `=`, `:=` in their respective contexts

`-Xfatal-warnings` is active. All warnings are errors.

### Import ordering

Three groups, each separated by a blank line, in this order:

```scala
// 1. Project-internal
import io.github.semyonsinchenko.safetensors.core.{SafetensorsDtype, TensorSchema}
import io.github.semyonsinchenko.safetensors.util.Errors

// 2. Spark / Hadoop / Jackson (provided dependencies)
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

// 3. Java / Scala stdlib
import java.nio.{ByteBuffer, ByteOrder}
import scala.jdk.CollectionConverters._
```

### Naming conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Classes, objects, traits | `PascalCase` | `SafetensorsTableProvider` |
| Methods, `val`, `var`, parameters | `camelCase` | `headerSize`, `buildForBatch` |
| Module-levelPPER_SNAKE constants | `U_CASE` | `DATA_FIELD`, `MIN_SHARD_SIZE_MB` |
| Type parameters | Single uppercase letter | `T`, `K`, `V` |
| Private helpers in objects | `camelCase` with `private` modifier | `private def parseJson(...)` |

### `null` and `Option`

Never return or accept `null` in Scala code. Wrap Java APIs:

```scala
// correct
Option(options.get("dtype")).map(...)

// wrong — never do this
val dtype = options.get("dtype")  // might be null
```

Use `Option[T]` in all Scala signatures. Use `None` / `getOrElse` / `fold`
rather than `.isEmpty` / `.get`.

### Collections

- Use `Seq[T]` (immutable) in all public signatures.
- Use `Map[K, V]` (immutable) for mappings.
- Use `Array[T]` only for JVM interop (`InternalRow`, `Array[Byte]`, Spark
  partition arrays).
- Prefer `.view.filterKeys(...).toMap` over `.filterKeys(...)` directly
  (deprecated in 2.13).

### Error handling

| Situation | Pattern |
|-----------|---------|
| Analysis-time (plan validation) | `throw Errors.analysisException(message)` |
| Precondition / programming error | `require(cond, message)` or `throw new IllegalArgumentException(msg)` |
| Expected failure with caller choice | Return `Either[String, T]` — see `SafetensorsDtype.fromString` |
| Catching exceptions | `case NonFatal(e) =>` — never catch `Exception` or `Throwable` broadly |

**Never** call `new AnalysisException(message)` directly — Spark 4 removed
the single-String constructor. Always use `Errors.analysisException()`.

### Scaladoc

- **Class/object level:** required for all public types. Describe purpose,
  not implementation. Cross-reference `format/SPECIFICATION.md` sections for
  architecture context.
- **Method level:** required for public methods with non-obvious behaviour.
  Use `@param`, `@return`, `@throws`.
- **Private helpers:** inline `//` comments are sufficient.
- Format: `/** ... */` (not `/* */`).

---

## 4. Python Code Style

### Imports

```python
from __future__ import annotations   # always first

# stdlib
import json
import os
from pathlib import Path
from typing import Optional

# third-party (heavy imports deferred inside functions to avoid import-time failures)
# import mlflow  ← inside function body, not at module level
```

### Type hints

All public function signatures must carry type hints. Use `Optional[T]` (not
`T | None`) for Python 3.10 compatibility.

```python
def log_dataset(
    path: str,
    run_id: Optional[str] = None,
    name: str = "safetensors_dataset",
) -> None:
```

### Docstrings

NumPy style with `Parameters`, `Returns`, `Raises` sections:

```python
"""Short summary.

Parameters
----------
path:
    Description.
run_id:
    Description.

Raises
------
FileNotFoundError
    When the manifest is absent.
"""
```

### Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Private helpers: `_leading_underscore`
- Module-level constants: `UPPER_SNAKE_CASE`

---

## 5. Architectural Invariants

These decisions must not be reversed without updating `format/SPECIFICATION.md`:

| Invariant | Rationale |
|-----------|-----------|
| BF16 is hardcoded as a valid dtype, bypassing the JSON schema regex | The schema pattern `([UIF])(8|16|32|64)` does not match `BF16`; this is a known quirk of the safetensors library. Every acceptance site must carry a `// NOTE: BF16 is not in the JSON schema regex` comment. |
| `Errors.analysisException()` is the only way to create `AnalysisException` | Spark 4 removed the `AnalysisException(String)` constructor. Direct `new AnalysisException(message)` will not compile. |
| Write path I/O goes through Hadoop `FileSystem` API | Required for HDFS, S3, GCS, Azure Blob compatibility. Never use `java.io.File` or `java.nio.file.Files` for output. |
| One `.safetensors` file = one `InputPartition`; files are never split | The full header must be read before locating any tensor; mid-file splits are not possible. |
| Tensor bytes must never be heap-copied as a staging buffer | Read path: slice `MappedByteBuffer`. Write path: `ByteBuffer.wrap(bytes)` directly. |
| Schema and type validation belongs in `WriteBuilder.buildForBatch()` | Errors must surface at plan time, before any tasks are launched. `DataWriter.write()` must not perform schema validation. |
| Global index file is `_tensor_index.parquet` | `_metadata.parquet` is reserved by Spark's partition discovery and must not be used. |
