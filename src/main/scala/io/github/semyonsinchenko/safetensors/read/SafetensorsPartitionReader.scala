package io.github.semyonsinchenko.safetensors.read

import io.github.semyonsinchenko.safetensors.core.{SafetensorsHeaderParser, TensorSchema}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.connector.read.PartitionReader
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.unsafe.types.UTF8String

import java.net.URI
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Paths, StandardOpenOption}

/** Reads one .safetensors file as a single InternalRow per file.
  *
  * Memory model (read path):
  *   - For local paths: the file is mapped into off-heap memory via FileChannel.map() →
  *     MappedByteBuffer. No heap copy of tensor bytes.
  *   - For remote paths (HDFS, S3, GCS, …): the file is read via Hadoop FileSystem into a heap
  *     ByteBuffer. mmap is only available locally.
  *   - Each tensor's bytes are extracted from the buffer by slicing.
  *
  * KNOWN LIMITATION: MappedByteBuffer cannot be explicitly unmapped before GC on JVM < 21. On JVM
  * 17 we use sun.misc.Unsafe.invokeCleaner as a workaround. See sun.misc.Cleaner usage in close().
  *
  * Column pruning: if a tensor column's "data" sub-field is not in the projected schema, its byte
  * buffer is skipped (shape and dtype still come from header).
  *
  * File size limitation (local mmap path): offsets are stored as Long but
  * MappedByteBuffer.position() takes an Int. Files ≥ 2 GB on the local path will throw an
  * IllegalArgumentException with a clear message directing users to use a remote filesystem path
  * for very large files.
  */
class SafetensorsPartitionReader(
    private val filePath: String,
    private val schema: StructType,
    private val options: CaseInsensitiveStringMap,
    private val hadoopConf: Configuration
) extends PartitionReader[InternalRow] {

  // Each .safetensors file produces exactly one row.
  private var rowEmitted = false

  // Local mmap path
  private var mappedBuffer: java.nio.MappedByteBuffer = _
  private var channel: FileChannel                    = _

  // Remote / heap path
  private var heapBuffer: ByteBuffer = _

  // Tracks which buffer to use
  private var useHeap: Boolean = false

  override def next(): Boolean =
    if (!rowEmitted) {
      openFile()
      true
    } else {
      false
    }

  override def get(): InternalRow = {
    val buf = if (useHeap) heapBuffer else mappedBuffer
    require(buf != null, "next() must be called before get()")
    rowEmitted = true
    buildRow(buf)
  }

  override def close(): Unit = {
    heapBuffer = null
    if (channel != null) {
      channel.close()
      channel = null
    }
    if (mappedBuffer != null) {
      tryUnmap(mappedBuffer)
      mappedBuffer = null
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def isLocalPath(path: String): Boolean = {
    val uri    = URI.create(path)
    val scheme = uri.getScheme
    scheme == null || scheme == "file"
  }

  private def openFile(): Unit =
    if (isLocalPath(filePath)) {
      openLocalFile()
    } else {
      openRemoteFile()
    }

  private def openLocalFile(): Unit = {
    // Strip leading "file://" if present so Paths.get() works correctly.
    val localPath =
      if (filePath.startsWith("file://")) filePath.drop(7)
      else if (filePath.startsWith("file:")) filePath.drop(5)
      else filePath

    channel = FileChannel.open(Paths.get(localPath), StandardOpenOption.READ)
    mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    useHeap = false
  }

  private def openRemoteFile(): Unit = {
    val hadoopPath = new Path(filePath)
    val fs         = FileSystem.get(hadoopPath.toUri, hadoopConf)
    val fileLen    = fs.getFileStatus(hadoopPath).getLen

    // Read entire file into a heap ByteBuffer. For remote files mmap is not
    // available; the Tensor bytes must be on the heap for this path.
    val bytes = new Array[Byte](fileLen.toInt)
    val in    = fs.open(hadoopPath)
    try {
      var offset = 0
      while (offset < bytes.length) {
        val n = in.read(bytes, offset, bytes.length - offset)
        if (n == -1) throw new java.io.EOFException(s"Unexpected EOF reading $filePath")
        offset += n
      }
    } finally
      in.close()

    heapBuffer = ByteBuffer.wrap(bytes)
    useHeap = true
  }

  private def buildRow(buf: ByteBuffer): InternalRow = {
    buf.rewind()
    val header = SafetensorsHeaderParser.parse(buf)

    val fields = schema.fields.map { field =>
      val tensorName = field.name
      header.tensors.get(tensorName) match {
        case None =>
          // Column exists in schema but not in this file — return null struct
          null
        case Some(info) =>
          val byteBufferStart: Long = header.byteBufferOffset

          // Read data bytes only if the "data" sub-field is in the projected schema
          val needData = field.dataType match {
            case st: StructType => st.fieldNames.contains(TensorSchema.DATA_FIELD)
            case _              => false
          }

          val dataBytes: Array[Byte] =
            if (needData) {
              val absBegin: Long = byteBufferStart + info.dataOffsets.begin
              val len: Int       = info.dataOffsets.byteSize.toInt

              require(
                absBegin <= Int.MaxValue,
                s"Tensor '$tensorName' starts at byte offset $absBegin which exceeds " +
                  s"the 2 GB MappedByteBuffer limit on the local-file read path. " +
                  s"Use a remote filesystem URI (hdfs://, s3a://, etc.) for files larger than 2 GB."
              )

              val bytes = new Array[Byte](len)
              val slice = buf.duplicate()
              slice.position(absBegin.toInt)
              slice.get(bytes)
              bytes
            } else {
              Array.emptyByteArray
            }

          // Build the Tensor Struct InternalRow
          val shapeArray = new GenericArrayData(info.shape.map(_.asInstanceOf[AnyRef]).toArray)
          val row        = new GenericInternalRow(3)
          row.update(0, dataBytes)                              // data:  BinaryType
          row.update(1, shapeArray)                             // shape: ArrayType(IntegerType)
          row.update(2, UTF8String.fromString(info.dtype.name)) // dtype: StringType
          row
      }
    }

    new GenericInternalRow(fields.asInstanceOf[Array[Any]])
  }

  /** Best-effort eager MappedByteBuffer unmap using sun.misc.Unsafe (JDK 9+). */
  private def tryUnmap(buf: java.nio.MappedByteBuffer): Unit =
    try {
      val unsafeClass = Class.forName("sun.misc.Unsafe")
      val theUnsafe   = unsafeClass.getDeclaredField("theUnsafe")
      theUnsafe.setAccessible(true)
      val unsafe = theUnsafe.get(null)
      val method = unsafeClass.getMethod("invokeCleaner", classOf[java.nio.ByteBuffer])
      method.invoke(unsafe, buf)
    } catch {
      case _: Exception => // Ignore — GC will unmap eventually
    }

}
