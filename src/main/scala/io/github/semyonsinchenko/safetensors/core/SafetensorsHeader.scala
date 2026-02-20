package io.github.semyonsinchenko.safetensors.core

/**
 * In-memory representation of a parsed safetensors file header.
 *
 * Binary layout (from format/format.md):
 *   - 8 bytes:  N (unsigned little-endian 64-bit integer) = size of JSON header
 *   - N bytes:  UTF-8 JSON string (must start with '{', may be space-padded)
 *   - rest:     byte buffer containing all tensor data
 *
 * The JSON header is a map from tensor name to TensorInfo, with the optional
 * reserved key "__metadata__" for free-form string-to-string metadata.
 *
 * data_offsets are relative to the start of the byte buffer (i.e. NOT to the
 * start of the file). Absolute file offset of tensor data =
 *   8 + N + data_offsets.begin
 *
 * The byte buffer must be fully indexed (no holes between tensors).
 * Tensor values are stored in little-endian, C/row-major order.
 */

/** Offset pair for a tensor's data within the byte buffer. */
final case class DataOffsets(begin: Long, end: Long) {
  /** Total byte size of this tensor. */
  def byteSize: Long = end - begin
}

/** Per-tensor metadata as parsed from the safetensors JSON header. */
final case class TensorInfo(
  dtype:       SafetensorsDtype,
  shape:       Seq[Int],
  dataOffsets: DataOffsets,
)

/**
 * Parsed representation of a complete safetensors file header.
 *
 * @param tensors   Ordered map from tensor name to TensorInfo.
 *                  Ordering follows the original JSON key order.
 * @param metadata  Optional free-form string-to-string metadata
 *                  (from the "__metadata__" key).
 * @param headerSize Size of the JSON header in bytes (= N from the 8-byte prefix).
 *                   The byte buffer starts at file offset 8 + headerSize.
 */
final case class SafetensorsHeader(
  tensors:    Map[String, TensorInfo],
  metadata:   Map[String, String],
  headerSize: Long,
) {
  /** Absolute file offset of the start of the byte buffer. */
  def byteBufferOffset: Long = 8L + headerSize

  /** Tensor names in insertion order. */
  def tensorNames: Seq[String] = tensors.keys.toSeq
}
