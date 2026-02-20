package io.github.semyonsinchenko.safetensors.core

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.json.JsonMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.util.{LinkedHashMap => JLinkedHashMap}

/** Serialises an in-memory tensor layout to a safetensors binary header.
  *
  * Produces:
  *   - 8 bytes: N (unsigned LE 64-bit) = byte length of JSON string
  *   - N bytes: UTF-8 JSON header (starts with '{', no padding added)
  *
  * The caller is responsible for writing the byte buffer (tensor data) after the header.
  * data_offsets in the JSON are relative to the start of the byte buffer, NOT to the start of the
  * file.
  *
  * Constraints from format/format.md:
  *   - No duplicate keys.
  *   - The byte buffer must be fully indexed (no holes).
  *   - Tensor data is little-endian, C/row-major order.
  */
object SafetensorsHeaderWriter {

  // Use JsonMapper.builder() so that the mapper serializes Java LinkedHashMap
  // in insertion order, preserving the tensor key ordering in the JSON header.
  private val mapper: ObjectMapper = JsonMapper
    .builder()
    .addModule(DefaultScalaModule)
    .build()

  /** Descriptor for one tensor to be written.
    *
    * @param name
    *   Tensor key in the safetensors header.
    * @param dtype
    *   Dtype string (e.g. "F32", "BF16").
    * @param shape
    *   Dimension sizes.
    * @param byteLength
    *   Exact number of bytes for this tensor's data.
    */
  final case class TensorDescriptor(
      name: String,
      dtype: SafetensorsDtype,
      shape: Seq[Int],
      byteLength: Long
  )

  /** Build a complete binary safetensors header for the given tensors.
    *
    * Tensors are written in the order supplied. The byte buffer starts immediately after the
    * header; data_offsets are computed accordingly.
    *
    * @param tensors
    *   Ordered sequence of tensor descriptors.
    * @param fileMetadata
    *   Optional __metadata__ key-value pairs.
    * @return
    *   ByteBuffer containing the 8-byte length prefix + JSON bytes, ready to be written to a file.
    */
  def buildHeader(
      tensors: Seq[TensorDescriptor],
      fileMetadata: Map[String, String] = Map.empty
  ): ByteBuffer = {
    val jsonStr    = buildJson(tensors, fileMetadata)
    val jsonBytes  = jsonStr.getBytes(StandardCharsets.UTF_8)
    val headerSize = jsonBytes.length.toLong

    // 8-byte LE prefix + JSON bytes
    val buf = ByteBuffer.allocate(8 + jsonBytes.length)
    buf.order(ByteOrder.LITTLE_ENDIAN)
    buf.putLong(headerSize)
    buf.put(jsonBytes)
    buf.flip()
    buf
  }

  private def buildJson(
      tensors: Seq[TensorDescriptor],
      fileMetadata: Map[String, String]
  ): String = {
    // Validate no duplicate names
    val names = tensors.map(_.name)
    val dupes = names.diff(names.distinct)
    require(dupes.isEmpty, s"Duplicate tensor names in header: ${dupes.mkString(", ")}")

    // Build the header as a Java LinkedHashMap to guarantee that Jackson
    // serialises keys in insertion order (tensor key order must match the
    // byte buffer layout so that data_offsets are correct).
    val headerMap = new JLinkedHashMap[String, Any]()

    // __metadata__ is written first when present
    if (fileMetadata.nonEmpty) {
      headerMap.put("__metadata__", fileMetadata)
    }

    // Compute data_offsets (byte buffer is fully indexed, no holes)
    var offset = 0L
    tensors.foreach { td =>
      val begin = offset
      val end   = offset + td.byteLength
      offset = end

      val tensorEntry = new JLinkedHashMap[String, Any]()
      tensorEntry.put("dtype", td.dtype.name)
      tensorEntry.put("shape", td.shape)
      tensorEntry.put("data_offsets", Seq(begin, end))
      headerMap.put(td.name, tensorEntry)
    }

    mapper.writeValueAsString(headerMap)
  }

}
