package io.github.semyonsinchenko.safetensors.core

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.json.JsonMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.immutable.ListMap

/** Parses the binary safetensors header from a ByteBuffer positioned at the start of the file.
  *
  * Format (from format/format.md):
  *   - 8 bytes: N, unsigned LE 64-bit integer = byte length of JSON header
  *   - N bytes: UTF-8 JSON string (starts with '{', may be space-padded)
  *   - rest: raw tensor byte buffer
  *
  * This parser reads only the 8 + N header bytes. Tensor data bytes are NOT read here — callers use
  * the DataOffsets from TensorInfo to locate them.
  */
object SafetensorsHeaderParser {

  // Configure Jackson to preserve JSON key insertion order by using
  // LinkedHashMap for object deserialization. This is required so that
  // tensor keys in SafetensorsHeader.tensors reflect the original file order.
  private val mapper = JsonMapper
    .builder()
    .addModule(DefaultScalaModule)
    .build()

  /** Parse a safetensors header from the given ByteBuffer.
    *
    * The buffer must be positioned at the very start of the file (offset 0) and must have at least
    * 8 bytes available. The buffer's byte order will be set to LITTLE_ENDIAN during parsing.
    *
    * @param buf
    *   ByteBuffer backed by the file (e.g. a MappedByteBuffer).
    * @return
    *   Parsed SafetensorsHeader.
    * @throws IllegalArgumentException
    *   if the header is malformed.
    */
  def parse(buf: ByteBuffer): SafetensorsHeader = {
    buf.order(ByteOrder.LITTLE_ENDIAN)

    require(buf.remaining() >= 8, "File too small to contain a safetensors header (< 8 bytes)")

    // Read the 8-byte header-length prefix
    val headerSize = buf.getLong()
    require(
      headerSize > 0 && headerSize <= Int.MaxValue,
      s"Safetensors header size out of range: $headerSize"
    )

    val jsonBytes = headerSize.toInt
    require(
      buf.remaining() >= jsonBytes,
      s"File too small: header claims $jsonBytes bytes but only ${buf.remaining()} remain"
    )

    // Extract JSON bytes without heap copy
    val jsonBuf = new Array[Byte](jsonBytes)
    buf.get(jsonBuf)

    // Trim trailing whitespace padding (the format allows 0x20 padding)
    val jsonStr = new String(jsonBuf, "UTF-8").trim

    require(
      jsonStr.startsWith("{"),
      s"Safetensors JSON header must start with '{', got: ${jsonStr.take(20)}"
    )

    parseJson(jsonStr, headerSize)
  }

  private def parseJson(json: String, headerSize: Long): SafetensorsHeader = {
    // Jackson with JsonMapper.builder() uses LinkedHashMap for objects,
    // which preserves JSON key insertion order.
    val root = mapper.readValue(json, classOf[Map[String, Any]])

    val metadata: Map[String, String] = root.get("__metadata__") match {
      case Some(m: Map[_, _]) => m.asInstanceOf[Map[String, String]]
      case _                  => Map.empty
    }

    // Build a ListMap to preserve the original JSON tensor key order.
    // ListMap.from preserves the iteration order of the input entries.
    val tensorEntries = root.view
      .filterKeys(_ != "__metadata__")
      .map { case (name, value) =>
        val tensorMap = value.asInstanceOf[Map[String, Any]]

        val dtypeStr = tensorMap("dtype").asInstanceOf[String]
        val dtype    = SafetensorsDtype.fromStringUnsafe(dtypeStr)

        val shape: Seq[Int] = tensorMap("shape") match {
          case list: Seq[_] =>
            list.map {
              case n: Int    => n
              case n: Long   => n.toInt
              case n: Number => n.intValue()
            }
          case other =>
            throw new IllegalArgumentException(
              s"Unexpected shape type for tensor '$name': ${other.getClass}"
            )
        }

        val offsets = tensorMap("data_offsets") match {
          case list: Seq[_] if list.length == 2 =>
            val begin = list.head match {
              case n: Int    => n.toLong
              case n: Long   => n
              case n: Number => n.longValue()
            }
            val end = list(1) match {
              case n: Int    => n.toLong
              case n: Long   => n
              case n: Number => n.longValue()
            }
            DataOffsets(begin, end)
          case other =>
            throw new IllegalArgumentException(
              s"Unexpected data_offsets type for tensor '$name': ${other.getClass}"
            )
        }

        name -> TensorInfo(dtype, shape, offsets)
      }
      .toSeq // materialize in order before building ListMap

    val tensors: ListMap[String, TensorInfo] = ListMap.from(tensorEntries)

    SafetensorsHeader(tensors, metadata, headerSize)
  }

}
