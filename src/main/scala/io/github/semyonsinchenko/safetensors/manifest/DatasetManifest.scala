package io.github.semyonsinchenko.safetensors.manifest

import com.fasterxml.jackson.annotation.JsonProperty

/** In-memory representation of dataset_manifest.json.
  *
  * Written by BatchWrite.commit() after all tasks complete. Structure per project specification
  * §3.5:
  *
  * { "format_version": "1.0", "safetensors_version": "1.0", "total_samples": 1000000,
  * "total_bytes": 1024000000, "shards": [ { "shard_path": "part-00000-<uuid>.safetensors",
  * "samples_count": 1000, "bytes": 1024000 } ], "schema": { "tensor_name": { "dtype": "F32",
  * "shape": [1000, 3, 224, 224] } } }
  */
final case class DatasetManifest(
    @JsonProperty("format_version") formatVersion: String,
    @JsonProperty("safetensors_version") safetensorsVersion: String,
    @JsonProperty("total_samples") totalSamples: Long,
    @JsonProperty("total_bytes") totalBytes: Long,
    @JsonProperty("shards") shards: Seq[ShardInfo],
    @JsonProperty("schema") schema: Map[String, TensorSchemaInfo]
)

final case class ShardInfo(
    @JsonProperty("shard_path") shardPath: String,
    @JsonProperty("samples_count") samplesCount: Int,
    @JsonProperty("bytes") bytes: Long
)

final case class TensorSchemaInfo(
    @JsonProperty("dtype") dtype: String,
    @JsonProperty("shape") shape: Seq[Int]
)

/** One entry in the global _tensor_index.parquet written when generate_index=true. Each row maps
  * one tensor key in one shard file to its shape and dtype.
  *
  * See §3.6 of the project specification.
  */
final case class TensorIndexEntry(
    tensorKey: String,
    fileName: String,
    shape: Seq[Int],
    dtype: String
)
