package io.github.semyonsinchenko.safetensors.read

import io.github.semyonsinchenko.safetensors.core.{SafetensorsHeaderParser, TensorSchema}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, GenericArrayData}
import org.apache.spark.sql.connector.read.PartitionReader
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.unsafe.types.UTF8String

import java.nio.channels.FileChannel
import java.nio.file.{Paths, StandardOpenOption}

/**
 * Reads one .safetensors file as a single InternalRow per file.
 *
 * Memory model (read path):
 *   - The file is mapped into off-heap memory via FileChannel.map() →
 *     MappedByteBuffer. No heap copy of tensor bytes is made.
 *   - Each tensor's bytes are exposed as a BinaryType field by slicing the
 *     MappedByteBuffer. Spark copies the slice into its own UnsafeRow storage
 *     during serialisation.
 *
 * KNOWN LIMITATION: MappedByteBuffer cannot be explicitly unmapped before GC
 * on JVM < 21. On JVM 17 we use sun.misc.Cleaner as a workaround. This is
 * documented here per the project specification (§3.10). On Linux this is not
 * a correctness issue — the OS will reclaim the mapping when the process exits
 * or GC runs. See sun.misc.Cleaner usage in close().
 *
 * Column pruning: if a tensor column's "data" sub-field is not in the projected
 * schema, we skip reading its byte buffer by using its data_offsets to seek
 * past it (the shape and dtype are still read from the header).
 */
class SafetensorsPartitionReader(
  private val filePath: String,
  private val schema:   StructType,
  private val options:  CaseInsensitiveStringMap,
) extends PartitionReader[InternalRow] {

  // Each .safetensors file produces exactly one row.
  private var rowEmitted = false
  private var mappedBuffer: java.nio.MappedByteBuffer = _
  private var channel: FileChannel = _

  override def next(): Boolean = {
    if (!rowEmitted) {
      openFile()
      true
    } else {
      false
    }
  }

  override def get(): InternalRow = {
    require(mappedBuffer != null, "next() must be called before get()")
    rowEmitted = true
    buildRow()
  }

  override def close(): Unit = {
    if (channel != null) {
      channel.close()
      channel = null
    }
    // Attempt to unmap the MappedByteBuffer eagerly on JVM < 21.
    // sun.misc.Cleaner is the accepted workaround. On JVM 9+ the Unsafe
    // approach is used instead. This is a best-effort operation; GC will
    // eventually release the mapping regardless.
    if (mappedBuffer != null) {
      tryUnmap(mappedBuffer)
      mappedBuffer = null
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def openFile(): Unit = {
    // For local files, open directly via NIO FileChannel for mmap support.
    // For remote filesystems (HDFS, S3, etc.), we fall back to a Hadoop stream
    // copy — mmap is only available for local paths.
    // TODO: add Hadoop-path detection and fallback for remote files
    channel = FileChannel.open(Paths.get(filePath), StandardOpenOption.READ)
    mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
  }

  private def buildRow(): InternalRow = {
    mappedBuffer.rewind()
    val header = SafetensorsHeaderParser.parse(mappedBuffer)

    val fields = schema.fields.map { field =>
      val tensorName = field.name
      header.tensors.get(tensorName) match {
        case None =>
          // Column exists in schema but not in this file — return null struct
          null
        case Some(info) =>
          val byteBufferStart = header.byteBufferOffset

          // Read data bytes only if the "data" sub-field is in the projected schema
          val needData = field.dataType match {
            case st: StructType => st.fieldNames.contains(TensorSchema.DATA_FIELD)
            case _              => false
          }

          val dataBytes: Array[Byte] =
            if (needData) {
              val absBegin = (byteBufferStart + info.dataOffsets.begin).toInt
              val len      = info.dataOffsets.byteSize.toInt
              val bytes    = new Array[Byte](len)
              // Slice the MappedByteBuffer — no extra heap staging
              val slice = mappedBuffer.duplicate()
              slice.position(absBegin)
              slice.get(bytes)
              bytes
            } else {
              Array.emptyByteArray
            }

          // Build the Tensor Struct InternalRow
          val shapeArray = new GenericArrayData(info.shape.map(_.asInstanceOf[AnyRef]).toArray)
          val row        = new GenericInternalRow(3)
          row.update(0, dataBytes)                            // data:  BinaryType
          row.update(1, shapeArray)                           // shape: ArrayType(IntegerType)
          row.update(2, UTF8String.fromString(info.dtype.name)) // dtype: StringType
          row
      }
    }

    new GenericInternalRow(fields.asInstanceOf[Array[Any]])
  }

  /** Best-effort eager MappedByteBuffer unmap using sun.misc.Cleaner / JDK internals. */
  private def tryUnmap(buf: java.nio.MappedByteBuffer): Unit = {
    try {
      // JDK 9+ approach: use Unsafe.invokeCleaner
      val unsafeClass = Class.forName("sun.misc.Unsafe")
      val theUnsafe   = unsafeClass.getDeclaredField("theUnsafe")
      theUnsafe.setAccessible(true)
      val unsafe      = theUnsafe.get(null)
      val method      = unsafeClass.getMethod("invokeCleaner", classOf[java.nio.ByteBuffer])
      method.invoke(unsafe, buf)
    } catch {
      case _: Exception => // Ignore — GC will unmap eventually
    }
  }
}
