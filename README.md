# safetensors-spark

Apache Spark DataSource V2 connector for reading and writing
[Hugging Face safetensors](https://github.com/huggingface/safetensors) files
at scale. Designed for training data preparation and static feature stores,
with zero-copy read compatibility with PyTorch/Horovod.

---

## Requirements

| Component | Version |
|-----------|---------|
| Apache Spark | 4.0+ |
| Java | 11 or 17 |
| Scala | 2.13 (inferred from Spark version) |
| Python (integration tests / MLflow utility) | 3.10+ |

---

## Installation

Attach the fat JAR when submitting a Spark job:

```bash
spark-submit --jars safetensors-spark-assembly-<version>.jar ...
```

Or in a SparkSession:

```python
spark = (
    SparkSession.builder
    .config("spark.jars", "/path/to/safetensors-spark-assembly-<version>.jar")
    .getOrCreate()
)
```

Build the JAR from source:

```bash
sbt assembly
# target/scala-2.13/safetensors-spark-assembly-<version>.jar
```

---

## Reading safetensors files

The connector uses a **wide/columnar schema**: each tensor key becomes one
Spark column of type `struct<data: binary, shape: array<int>, dtype: string>`.

### Schema inference (convenience, slower)

```python
df = (
    spark.read
    .format("safetensors")
    .option("inferSchema", "true")
    .load("/data/tensors/")
)
df.printSchema()
# root
#  |-- image: struct (nullable = false)
#  |    |-- data:  binary    (nullable = false)
#  |    |-- shape: array<int>(nullable = false)
#  |    |-- dtype: string    (nullable = false)
#  |-- label: struct (nullable = false)
#  |    |-- ...
```

Schema is inferred from the header of the first file. All files in the dataset
are assumed to share the same schema.

### Explicit schema (recommended for production)

```python
from pyspark.sql.types import *

tensor = StructType([
    StructField("data",  BinaryType(),                    False),
    StructField("shape", ArrayType(IntegerType(), False), False),
    StructField("dtype", StringType(),                    False),
])

schema = StructType([
    StructField("image", tensor, False),
    StructField("label", tensor, False),
])

df = (
    spark.read
    .format("safetensors")
    .schema(schema)
    .load("/data/tensors/")
)
```

---

## Writing safetensors files

Every write produces `dataset_manifest.json` at the output root. There are two
write modes, selected by mutually exclusive options.

### Batch mode (`batch_size`)

Groups exactly N input rows into one standalone safetensors file per batch. Every
output file contains one tensor per column with shape `[batch_size, *per_sample_shape]`,
ready for direct GPU loading without any further preprocessing.

```python
(
    df.write
    .format("safetensors")
    .option("batch_size", "1000")   # 1000 rows → one tensor per column per file
    .option("dtype", "F32")
    .mode("overwrite")
    .save("/output/tensors/")
)
```

With explicit per-sample shape and a custom tail strategy:

```python
import json

(
    df.write
    .format("safetensors")
    .option("batch_size", "512")
    .option("dtype", "F16")
    .option("shapes", json.dumps({"image": [3, 224, 224], "label": []}))
    .option("tail_strategy", "pad")   # zero-pad the last batch to exactly 512 rows
    .mode("overwrite")
    .save("/output/tensors/")
)
# Each output file contains tensors shaped [512, 3, 224, 224] and [512].
# "shapes" specifies the per-sample shape; batch_size is prepended automatically.
```

Each Spark partition writes its batches independently. Repartition your DataFrame
before writing if you need balanced shard counts across partitions.

### KV-store mode (`name_col`)

Each input row produces one tensor per non-key column. The tensor key is
constructed as `{name_col_value}{kv_separator}{column_name}`. Suited for feature
stores where individual samples must be retrieved by key.

**Single tensor column:**

```python
# df has columns: user_id (string), embedding (Tensor Struct)
(
    df.write
    .format("safetensors")
    .option("name_col", "user_id")
    .option("dtype", "BF16")
    .option("generate_index", "true")   # write _tensor_index.parquet for fast lookups
    .mode("overwrite")
    .save("/output/embeddings/")
)
# Tensor keys: "user_42__embedding", "user_1__embedding", ...
```

**Multiple tensor columns:**

```python
# df has columns: model_id (string), weights (Tensor Struct), bias (Tensor Struct)
(
    df.write
    .format("safetensors")
    .option("name_col", "model_id")
    .option("kv_separator", "/")          # customize separator (default: "__")
    .option("generate_index", "true")
    .mode("overwrite")
    .save("/output/models/")
)
# Tensor keys: "model_v1/weights", "model_v1/bias", "model_v2/weights", ...
```

Handling duplicate keys (default is `"fail"`):

```python
.option("duplicatesStrategy", "lastWin")   # last row's tensor wins silently
```

---

## Write options reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | Int | — | **Batch mode.** Each N rows produce one output file with stacked tensors. Mutually exclusive with `name_col`. |
| `name_col` | String | — | **KV mode.** Column whose value becomes the tensor key. Mutually exclusive with `batch_size`. |
| `tail_strategy` | String | `"drop"` | **Batch mode.** How to handle the last incomplete batch when partition size is not a multiple of `batch_size`: `"drop"` (discard), `"pad"` (zero-pad to `batch_size`), `"write"` (write as-is). |
| `kv_separator` | String | `"__"` | Separator between `name_col` value and column name in compound tensor key (KV mode with multiple columns). |
| `dtype` | String | — | Target dtype for encoding: `F16`, `F32`, `F64`, `BF16`, `U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `U64`, `I64`. Required when input columns are numeric arrays. |
| `columns` | String | all | Comma-separated list of columns to serialize. Other columns are ignored. |
| `shapes` | JSON | — | Per-sample shape override, e.g. `'{"image": [3, 224, 224]}'`. In batch mode, `batch_size` is prepended automatically as the leading dimension. |
| `duplicatesStrategy` | String | `"fail"` | Duplicate key behaviour in KV mode: `"fail"` (raise) or `"lastWin"` (keep last). |
| `generate_index` | Boolean | `false` | Write `_tensor_index.parquet` at the output root for O(1) key lookups. |
| `target_shard_size_mb` | Int | `300` | **KV mode only.** Target shard file size in MB. Valid range: 50–1000. |

### Accepted input column types

The connector auto-detects and accepts two column types:

| Column type | Behaviour |
|-------------|-----------|
| `struct<data: binary, shape: array<int>, dtype: string>` | Raw bytes written directly. |
| `array<FloatType \| DoubleType \| IntegerType \| LongType \| ShortType \| ByteType>` | Elements encoded to `dtype`. Requires the `dtype` option. |

All type and schema errors are reported at plan time, before any task starts.

---

## Output layout

```
/output/path/
  part-00000-0000-<uuid>.safetensors      # shard files
  part-00001-0000-<uuid>.safetensors
  ...
  dataset_manifest.json                   # always written
  _tensor_index.parquet                   # only when generate_index=true
```

`dataset_manifest.json` structure:

```json
{
  "format_version": "1.0",
  "safetensors_version": "1.0",
  "total_samples": 50000,
  "total_bytes": 204800000,
  "shards": [
    {
      "file": "part-00000-0000-550e8400-e29b-41d4-a716-446655440000.safetensors",
      "samples_count": 1000,
      "bytes": 4096000
    }
  ]
}
```

---

## Catalyst SQL functions

Enable the built-in Catalyst expressions by adding the extensions class to
your Spark configuration:

```python
spark = (
    SparkSession.builder
    .config(
        "spark.sql.extensions",
        "io.github.semyonsinchenko.safetensors.SafetensorsExtensions",
    )
    .getOrCreate()
)
```

### `arr_to_st(arrayCol, shape, dtype)`

Converts a flat `ArrayType(FloatType)` column into a Tensor Struct, encoding
the float values as raw bytes in the specified dtype.

```python
from pyspark.sql.functions import array, lit, col

df_tensors = df.select(
    arr_to_st(
        col("embedding_floats"),         # ArrayType(FloatType)
        array(lit(1), lit(768)),         # shape [1, 768]
        lit("F32"),                      # dtype
    ).alias("embedding")
)
```

> **Warning — BF16/F16 approximation:** When `dtype="BF16"`, conversion uses
> truncation of the top 16 bits of the IEEE 754 Float32 bit pattern, not
> round-to-nearest-even. Results may differ from PyTorch/NumPy by up to 1 ULP.
> Pre-convert using PyTorch or NumPy if numerical exactness is required. The
> same applies to `dtype="F16"`.

### `st_to_array(tensorCol)`

Decodes a Tensor Struct's raw bytes into a flat `ArrayType(FloatType)`, row-
major, upcasting all element types to Float32.

```python
df_floats = df.select(
    st_to_array(col("image")).alias("image_floats")
)
```

> **Note:** Returns Float32 regardless of the original dtype. For BF16 → Float32
> this upcast is lossless. For lossless round-trips, use the raw `data` field
> of the Tensor Struct directly.

---

## MLflow lineage (Python utility)

Install the Python package:

```bash
pip install python/safetensors_spark/   # from source
```

After writing a dataset from Spark, log it as an MLflow dataset artifact:

```python
import mlflow
from safetensors_spark.mlflow import log_dataset

with mlflow.start_run():
    # ... your Spark write ...
    log_dataset(
        path="/output/tensors/",
        name="training_embeddings",
    )
```

The function reads `dataset_manifest.json` directly — it has no dependency on
Apache Spark or the Scala connector.

```python
def log_dataset(
    path: str,
    run_id: Optional[str] = None,   # uses active run if None
    name: str = "safetensors_dataset",
) -> None
```

Raises `FileNotFoundError` if `dataset_manifest.json` is absent. Raises
`RuntimeError` if `run_id` is `None` and no MLflow run is active.

---

## Limitations & known caveats

| Limitation | Detail |
|------------|--------|
| BF16/F16 write approximation | `arr_to_st()` uses truncation, not round-to-nearest-even. See warning above. |
| Schema inference reads one file | All files are assumed to share the schema of the first file. |
| No mid-file splits | One `.safetensors` file = one Spark task. Files are not splittable. |
| Off-heap mapping on JVM < 21 | `MappedByteBuffer` cannot be explicitly unmapped before GC on JVM 11/17; the OS reclaims the mapping on GC. Not a correctness issue on Linux. |
| `_metadata.parquet` reserved | The global tensor index uses `_tensor_index.parquet`. The name `_metadata.parquet` is reserved by Spark's partition discovery and is not used. |
