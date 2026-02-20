package io.github.semyonsinchenko.safetensors.manifest

import com.fasterxml.jackson.annotation.JsonProperty

/**
 * In-memory representation of dataset_manifest.json.
 *
 * Written by BatchWrite.commit() after all tasks complete.
 * Structure per project specification §3.5:
 *
 * {
 *   "format_version": "1.0",
 *   "safetensors_version": "1.0",
 *   "total_samples": 1000000,
 *   "total_bytes": 1024000000,
 *   "shards": [
 *     { "file": "part-00000-<uuid>.safetensors", "samples_count": 1000, "bytes": 1024000 }
 *   ]
 * }
 */
final case class DatasetManifest(
  @JsonProperty("format_version")      formatVersion:      String,
  @JsonProperty("safetensors_version") safetensorsVersion: String,
  @JsonProperty("total_samples")       totalSamples:       Long,
  @JsonProperty("total_bytes")         totalBytes:         Long,
  @JsonProperty("shards")              shards:             Seq[ShardInfo],
)

final case class ShardInfo(
  @JsonProperty("file")          file:         String,
  @JsonProperty("samples_count") samplesCount: Int,
  @JsonProperty("bytes")         bytes:        Long,
)
