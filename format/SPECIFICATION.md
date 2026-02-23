# Safetensors Format Specification

This document is the authoritative specification for the safetensors file format
and the safetensors-spark dataset layout. It is intended for anyone implementing
support for this format in any language or tool.

**For user-facing API and usage examples see `README.md`.**
**For agent/build reference see `AGENTS.md`.**

---

## 1. Safetensors Binary Format

The safetensors binary format is defined by the Hugging Face safetensors library.
The files in this directory are the ground-truth references.

### 1.1 Binary Layout

```
┌─────────────────────────────────────────────────────────────┐
│ 8 bytes: N — unsigned little-endian 64-bit integer         │
│          containing the size of the header in bytes        │
├─────────────────────────────────────────────────────────────┤
│ N bytes: UTF-8 JSON string representing the header         │
│          - MUST begin with '{' character (0x7B)            │
│          - MAY be trailing padded with whitespace (0x20)   │
├─────────────────────────────────────────────────────────────┤
│ Rest of file: raw tensor byte buffer                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Header JSON Structure

```json
{
  "tensor_name": {
    "dtype": "F32",
    "shape": [1, 16, 256],
    "data_offsets": [BEGIN, END]
  },
  "next_tensor_name": {...},
  "__metadata__": {
    "key": "value"
  }
}
```

- **`dtype`**: Data type string (see §1.3)
- **`shape`**: Array of dimension sizes. Empty array `[]` for 0-rank (scalar) tensors.
- **`data_offsets`**: Two-element array `[BEGIN, END]` specifying the tensor's byte
  range **relative to the beginning of the byte buffer** (NOT absolute file offsets).
  Byte size = `END - BEGIN`.
- **`__metadata__`**: Optional free-form string-to-string map.

### 1.3 Data Types

| dtype | Description | Bytes per element |
|-------|-------------|-------------------|
| `F16` | Half-precision float (IEEE 754) | 2 |
| `F32` | Single-precision float (IEEE 754) | 4 |
| `F64` | Double-precision float (IEEE 754) | 8 |
| `BF16` | Brain float16 (truncated F32) | 2 |
| `U8` | Unsigned 8-bit integer | 1 |
| `I8` | Signed 8-bit integer | 1 |
| `U16` | Unsigned 16-bit integer | 2 |
| `I16` | Signed 16-bit integer | 2 |
| `U32` | Unsigned 32-bit integer | 4 |
| `I32` | Signed 32-bit integer | 4 |
| `U64` | Unsigned 64-bit integer | 8 |
| `I64` | Signed 64-bit integer | 8 |

### 1.4 Format Constraints

- **No duplicate tensor keys** — all keys in the header must be unique.
- **No holes in byte buffer** — tensor data must be contiguous with no gaps.
- **Little-endian** byte order throughout.
- **Row-major (C) order** for multi-dimensional tensors.
- Empty tensors (any dimension = 0) are allowed — they store no data but retain
  shape in the header.

---

## 2. JSON Schemas

### 2.1 Safetensors Header Schema

See `safetensors.schema.json` (copied verbatim from the Hugging Face safetensors
repository).

**Note:** The schema's `dtype` pattern is `([UIF])(8|16|32|64|128|256)`, which does
**NOT** match `"BF16"`. BF16 is a known special case handled by the safetensors
library outside the schema. Every implementation must hardcode `BF16` as a valid
dtype.

### 2.2 Dataset Manifest Schema

See `manifest-jsonschema.json`. This schema defines the structure of
`dataset_manifest.json` written by the safetensors-spark connector.

```json
{
  "format_version": "1.0",
  "safetensors_version": "1.0",
  "total_samples": 1000000,
  "total_bytes": 1024000000,
  "shards": [
    {
      "shard_path": "part-00000-0000-550e8400-e29b-41d4-a716-446655440000.safetensors",
      "samples_count": 1000,
      "bytes": 4096000
    }
  ],
  "schema": {
    "tensor_name": {
      "dtype": "F32",
      "shape": [1000, 3, 224, 224]
    }
  }
}
```

### 2.3 Tensor Index Schema

When `generate_index=true`, the connector writes `_tensor_index.parquet` with:

| Column | Type | Description |
|--------|------|-------------|
| `tensor_key` | String | Tensor name as stored in safetensors header |
| `file_name` | String | Shard filename containing this tensor |
| `shape` | Array[Int] | Tensor shape |
| `dtype` | String | Tensor data type |

---

## 3. Output Layout

### 3.1 Directory Structure

```
/output/path/
  part-00000-0000-<uuid>.safetensors      # shard files
  part-00001-0000-<uuid>.safetensors
  ...
  dataset_manifest.json                   # always written
  _tensor_index.parquet                   # only when generate_index=true
```

### 3.2 Output File Naming

```
part-{taskId:05d}-{shardIndex:04d}-{uuid}.safetensors
```

Example: `part-00003-0001-550e8400-e29b-41d4-a716-446655440000.safetensors`

- **`taskId`**: Spark task ID (zero-padded to 5 digits)
- **`shardIndex`**: Per-task shard counter (zero-padded to 4 digits), incremented
  each time a new file is sealed
- **`uuid`**: UUID generated at DataWriter construction time for uniqueness during
  speculative execution retries

---

## 4. Write Modes

### 4.1 Batch Mode (`batch_size`)

- Every `batch_size` rows produce one standalone safetensors file
- Each output file contains one tensor per input column
- Tensor shape: `[batch_size, *per_sample_shape]`
- Tail handling strategies:
  - `drop`: discard incomplete last batch
  - `pad`: zero-pad to exactly `batch_size` rows
  - `write`: write partial batch as-is

### 4.2 KV Mode (`name_col`)

- Each input row produces one tensor per non-key column
- Tensor key: `{name_col_value}{kv_separator}{column_name}`
- Shards roll over when accumulated size exceeds `target_shard_size_mb`
- Duplicate key handling: `fail` (raise exception) or `lastWin` (keep last)

---

## 5. Known Quirks

### 5.1 BF16 Not in JSON Schema

The `safetensors.schema.json` does not recognize `"BF16"` as a valid dtype because
its regex pattern `([UIF])(8|16|32|64|128|256)` doesn't match it. This is a known
quirk of the official safetensors library. Implementations must hardcode `BF16`
as a valid dtype.

### 5.2 F16/BF16 Approximation

When encoding float values to F16 or BF16, the safetensors-spark connector uses
**truncation** of the IEEE 754 bit patterns, not round-to-nearest-even. This may
differ from PyTorch/NumPy by up to 1 ULP. For numerical precision requirements,
pre-convert using PyTorch or NumPy before writing.