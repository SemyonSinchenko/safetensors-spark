package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.core.{
  SafetensorsDtype,
  SafetensorsHeaderWriter,
  TensorSchema
}
import io.github.semyonsinchenko.safetensors.core.SafetensorsHeaderWriter.TensorDescriptor
import io.github.semyonsinchenko.safetensors.manifest.{ShardInfo, TensorIndexEntry}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.connector.write.{DataWriter, WriterCommitMessage}
import org.apache.spark.sql.types._

import java.nio.{ByteBuffer, ByteOrder}
import java.util.UUID

/** DataWriter for safetensors output. Runs on each executor task.
  *
  * Output file naming: part-{taskId:05d}-{uuid}.safetensors (per spec §3.8)
  *
  * A new shard file is opened when estimated bytes exceed target_shard_size_mb.
  *
  * Batch mode (batch_size): Rows are accumulated into a buffer. When the buffer reaches batch_size,
  * one safetensors file is written with one tensor per schema column. The leading dimension is the
  * batch size; subsequent dimensions come from the shapes option or are inferred from the first
  * row.
  *
  * KV mode (name_col): Each row is written as one tensor whose key is the value of the name_col
  * column. Duplicate keys are handled per the duplicatesStrategy option (fail or lastWin).
  * One shard file may contain multiple tensors from different rows.
  *
  * Write path memory model (§3.10):
  *   - Tensor bytes from InternalRow (BinaryType) are wrapped with ByteBuffer.wrap() — no extra
  *     heap copy.
  *   - Numeric array encoding uses ByteBuffer.allocateDirect() per element group.
  *   - Hadoop FSDataOutputStream is used for all output (supports HDFS, S3, GCS).
  */
class SafetensorsDataWriter(
    private val partitionId: Int,
    private val taskId: Long,
    private val schema: StructType,
    private val options: WriteOptions,
    private val outputPath: String
) extends DataWriter[InternalRow] {

  private val taskUuid = UUID.randomUUID().toString

  // Accumulated shard info for the commit message
  private val shards = scala.collection.mutable.ArrayBuffer.empty[ShardInfo]

  // Tensor index entries for the optional _tensor_index.parquet
  private val indexEntries = scala.collection.mutable.ArrayBuffer.empty[TensorIndexEntry]

  // Current shard state
  private var currentShardStream: org.apache.hadoop.fs.FSDataOutputStream = _
  private var currentShardPath: String                                    = _
  private var currentShardBytes: Long                                     = 0L
  private var currentShardSamples: Int                                    = 0

  // Tracks every shard file path opened by this writer for abort cleanup
  private val openedShardPaths = scala.collection.mutable.ArrayBuffer.empty[String]

  // Pending rows in the current batch (batch_size mode)
  private val batchBuffer = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  // Accumulated tensors in current shard (KV mode) — (descriptor, raw bytes) pairs
  private val kvTensorBuffer = scala.collection.mutable.ArrayBuffer.empty[(TensorDescriptor, Array[Byte])]

  // Tracked tensor keys in current shard (KV mode, for duplicate detection)
  private val currentShardTensorKeys = scala.collection.mutable.Set.empty[String]

  private lazy val hadoopConf =
    SparkSession.active.sparkContext.hadoopConfiguration

  // Columns to write (resolved once at construction time)
  private val columnsToWrite: Seq[(String, Int, DataType)] = {
    val colNames = options.columns.getOrElse {
      options.namingStrategy match {
        case NameColStrategy(col) => schema.fieldNames.filterNot(_ == col).toSeq
        case _                    => schema.fieldNames.toSeq
      }
    }
    colNames.flatMap { name =>
      val idx = schema.fieldIndex(name)
      Some((name, idx, schema.fields(idx).dataType))
    }
  }

  override def write(row: InternalRow): Unit =
    options.namingStrategy match {
      case BatchSizeStrategy(batchSize) =>
        batchBuffer += row.copy()
        if (batchBuffer.size >= batchSize) {
          flushBatch()
        }
      case NameColStrategy(_) =>
        writeKVRow(row)
    }

  override def commit(): WriterCommitMessage = {
    options.namingStrategy match {
      case _: BatchSizeStrategy if batchBuffer.nonEmpty => flushBatch()
      case _: NameColStrategy if kvTensorBuffer.nonEmpty =>
        flushKVShard() // Flushes and seals the shard
      case _ =>
    }
    closeShard()
    SafetensorsCommitMessage(shards.toSeq, indexEntries.toSeq)
  }

  override def abort(): Unit = {
    batchBuffer.clear()
    closeShard()
    // Delete all partial shard files written by this task
    openedShardPaths.foreach { p =>
      try {
        val path = new Path(p)
        val fs   = FileSystem.get(path.toUri, hadoopConf)
        fs.delete(path, false /* recursive */ )
      } catch {
        case _: Exception => // best effort; ignore failures during abort
      }
    }
  }

  override def close(): Unit = closeShard()

  // ---------------------------------------------------------------------------
  // Private helpers — batch mode
  // ---------------------------------------------------------------------------

  private def flushBatch(): Unit = {
    if (batchBuffer.isEmpty) return

    val batchSize = batchBuffer.size
    openShardIfNeeded()

    // Build tensor descriptors (name, dtype, shape, byteLength) and raw byte arrays
    val tensors: Seq[(TensorDescriptor, Array[Byte])] = columnsToWrite.map {
      case (colName, colIdx, colType) =>
        val dtype = options.dtype.getOrElse {
          // For TensorStruct input, read dtype from the first row
          colType match {
            case st: StructType if TensorSchema.isTensorStruct(st) =>
              val firstRow = batchBuffer.head
              val struct   = firstRow.getStruct(colIdx, 3)
              SafetensorsDtype.fromStringUnsafe(struct.getUTF8String(2).toString)
            case _ =>
              throw new IllegalStateException(
                s"dtype option is required for numeric array column '$colName'"
              )
          }
        }

        val (perSampleShape, rawBytes) = colType match {
          case st: StructType if TensorSchema.isTensorStruct(st) =>
            // Tensor Struct input: concatenate raw data bytes from each row
            val byteArrays = batchBuffer.map { row =>
              row.getStruct(colIdx, 3).getBinary(0)
            }.toArray

            // Shape: infer per-sample shape from first row's shape field
            val firstStruct = batchBuffer.head.getStruct(colIdx, 3)
            val shapeArr    = firstStruct.getArray(1)
            val perSample   = (0 until shapeArr.numElements()).map(i => shapeArr.getInt(i))

            val combined = concatByteArrays(byteArrays)
            (perSample, combined)

          case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
            // Numeric array input: encode all rows to target dtype
            val byteArrays = batchBuffer.map { row =>
              val arr = row.getArray(colIdx)
              encodeNumericArray(arr, at.elementType, dtype)
            }.toArray

            // Infer per-sample shape from shapes option or first row element count
            val perSample = options.shapes.get(colName) match {
              case Some(s) => s
              case None =>
                val n = batchBuffer.head.getArray(colIdx).numElements()
                Seq(n)
            }

            val combined = concatByteArrays(byteArrays)
            (perSample, combined)

          case other =>
            throw new IllegalStateException(
              s"Unexpected column type for '$colName': ${other.simpleString}"
            )
        }

        // Prepend batch_size as the leading dimension
        val fullShape  = Seq(batchSize) ++ perSampleShape
        val descriptor = TensorDescriptor(colName, dtype, fullShape, rawBytes.length.toLong)
        (descriptor, rawBytes)
    }

    // Write safetensors header
    val descriptors = tensors.map(_._1)
    val headerBuf   = SafetensorsHeaderWriter.buildHeader(descriptors)
    val headerBytes = headerBuf.remaining()
    writeBuffer(headerBuf)

    // Write tensor data in order
    val dataBytes = tensors.map { case (_, bytes) =>
      currentShardStream.write(bytes)
      bytes.length.toLong
    }.sum

    val writtenTotal = headerBytes.toLong + dataBytes
    currentShardBytes += writtenTotal
    currentShardSamples += batchSize

    // Collect index entries for this flush
    if (options.generateIndex) {
      val shardFileName = new Path(currentShardPath).getName
      tensors.foreach { case (desc, _) =>
        indexEntries += TensorIndexEntry(
          tensorKey = desc.name,
          fileName = shardFileName,
          shape = desc.shape,
          dtype = desc.dtype.name
        )
      }
    }

    batchBuffer.clear()
    maybeSealShard()
  }

  // ---------------------------------------------------------------------------
  // Private helpers — KV mode
  // ---------------------------------------------------------------------------

  private def writeKVRow(row: InternalRow): Unit = {
    val NameColStrategy(nameColName) = options.namingStrategy

    // Extract the base key from the name_col column
    val nameColIdx = schema.fieldIndex(nameColName)
    val nameColValue = row.getUTF8String(nameColIdx).toString

    // Process each non-key tensor column, emitting one tensor per column
    for ((colName, colIdx, colType) <- columnsToWrite) {
      // Construct compound tensor key: {name_col_value}{separator}{column_name}
      val tensorKey = s"$nameColValue${options.kvSeparator}$colName"

      // Build tensor descriptor and raw bytes for this column
      val (tensorDescriptor, tensorBytes) = writeKVTensor(row, tensorKey, colName, colIdx, colType)

      // Handle duplicate detection
      if (currentShardTensorKeys.contains(tensorKey)) {
        options.duplicatesStrategy match {
          case FailOnDuplicate =>
            throw new IllegalStateException(
              s"Duplicate tensor key '$tensorKey' in name_col mode with duplicatesStrategy=fail. " +
                s"Either use duplicatesStrategy=lastWin or ensure unique keys."
            )
          case LastWinOnDuplicate =>
            // Silently overwrite: remove the old one and add the new one
            kvTensorBuffer --= kvTensorBuffer.filter(_._1.name == tensorKey)
        }
      }

      currentShardTensorKeys += tensorKey
      kvTensorBuffer += ((tensorDescriptor, tensorBytes))

      currentShardSamples += 1
    }

    // Check shard size threshold after all columns for this row are processed
    val estimatedBytes = kvTensorBuffer.map(_._2.length.toLong).sum + kvTensorBuffer.length * 200L
    val thresholdBytes = options.targetShardSizeMb.toLong * 1024 * 1024
    if (estimatedBytes >= thresholdBytes) {
      flushKVShard()
    }
  }

  /** Flush accumulated KV tensors to a shard file. */
  private def flushKVShard(): Unit = {
    if (kvTensorBuffer.isEmpty) return

    openShardIfNeeded()

    // Build one safetensors file with all accumulated tensors
    val tensors = kvTensorBuffer.toSeq
    val headerBuf   = SafetensorsHeaderWriter.buildHeader(tensors.map(_._1))
    val headerBytes = headerBuf.remaining()
    writeBuffer(headerBuf)

    // Write all tensor data in order
    tensors.foreach { case (_, bytes) =>
      currentShardStream.write(bytes)
    }

    val dataBytes = tensors.map(_._2.length.toLong).sum
    currentShardBytes += headerBytes.toLong + dataBytes

    // Collect index entries
    if (options.generateIndex) {
      val shardFileName = new Path(currentShardPath).getName
      tensors.foreach { case (desc, _) =>
        indexEntries += TensorIndexEntry(
          tensorKey = desc.name,
          fileName = shardFileName,
          shape = desc.shape,
          dtype = desc.dtype.name
        )
      }
    }

    kvTensorBuffer.clear()
    currentShardTensorKeys.clear()
    sealCurrentShard()
  }

  /** Extract a single column from a row as a tensor and return (descriptor, rawBytes). */
  private def writeKVTensor(
      row: InternalRow,
      tensorKey: String,
      colName: String,
      colIdx: Int,
      colType: DataType
  ): (TensorDescriptor, Array[Byte]) = {
    val dtype = options.dtype.getOrElse {
      colType match {
        case st: StructType if TensorSchema.isTensorStruct(st) =>
          val struct = row.getStruct(colIdx, 3)
          SafetensorsDtype.fromStringUnsafe(struct.getUTF8String(2).toString)
        case _ =>
          throw new IllegalStateException(
            s"dtype option is required for numeric array column '$colName' in KV mode"
          )
      }
    }

    val (perSampleShape, bytes) = colType match {
      case st: StructType if TensorSchema.isTensorStruct(st) =>
        val struct = row.getStruct(colIdx, 3)
        val shapeArr = struct.getArray(1)
        val shape =
          (0 until shapeArr.numElements()).map(i => shapeArr.getInt(i)).toSeq
        val data = struct.getBinary(0)
        (shape, data)

      case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
        val arr = row.getArray(colIdx)
        val shape = options.shapes.get(colName) match {
          case Some(s) => s
          case None    => Seq(arr.numElements())
        }
        val bytes = encodeNumericArray(arr, at.elementType, dtype)
        (shape, bytes)

      case other =>
        throw new IllegalStateException(
          s"Unexpected column type for '$colName': ${other.simpleString}"
        )
    }

    (TensorDescriptor(tensorKey, dtype, perSampleShape, bytes.length.toLong), bytes)
  }

  // ---------------------------------------------------------------------------
  // Private helpers — encoding
  // ---------------------------------------------------------------------------

  /** Encode a numeric InternalRow ArrayData to raw little-endian bytes in the target safetensors
    * dtype.
    */
  private def encodeNumericArray(
      arr: ArrayData,
      sourceType: DataType,
      dtype: SafetensorsDtype
  ): Array[Byte] = {
    val nElem        = arr.numElements()
    val bytesPerElem = SafetensorsDtype.bytesPerElement(dtype)
    val buf          = ByteBuffer.allocateDirect(nElem * bytesPerElem)
    buf.order(ByteOrder.LITTLE_ENDIAN)

    for (i <- 0 until nElem) {
      // Read value as Double (widest numeric type) to avoid precision loss
      val v: Double = sourceType match {
        case FloatType   => arr.getFloat(i).toDouble
        case DoubleType  => arr.getDouble(i)
        case IntegerType => arr.getInt(i).toDouble
        case LongType    => arr.getLong(i).toDouble
        case ShortType   => arr.getShort(i).toDouble
        case ByteType    => arr.getByte(i).toDouble
        case other =>
          throw new IllegalArgumentException(
            s"Unsupported numeric array element type: ${other.simpleString}"
          )
      }

      dtype match {
        case SafetensorsDtype.F32 => buf.putFloat(v.toFloat)
        case SafetensorsDtype.F64 => buf.putDouble(v)
        case SafetensorsDtype.I8  => buf.put(v.toByte)
        case SafetensorsDtype.U8  => buf.put((v.toInt & 0xff).toByte)
        case SafetensorsDtype.I16 => buf.putShort(v.toShort)
        case SafetensorsDtype.U16 => buf.putShort((v.toInt & 0xffff).toShort)
        case SafetensorsDtype.I32 => buf.putInt(v.toInt)
        case SafetensorsDtype.U32 => buf.putInt(v.toLong.toInt)
        case SafetensorsDtype.I64 => buf.putLong(v.toLong)
        case SafetensorsDtype.U64 => buf.putLong(v.toLong)
        // BF16: truncate top 16 bits of Float32 IEEE 754 bit pattern.
        // NOTE: BF16 is not in the JSON schema regex — see §1.1.
        case SafetensorsDtype.BF16 =>
          val bits = java.lang.Float.floatToRawIntBits(v.toFloat)
          buf.putShort((bits >>> 16).toShort)
        // F16: approximate truncation (not round-to-nearest-even).
        case SafetensorsDtype.F16 =>
          buf.putShort(floatToFloat16Truncate(v.toFloat))
      }
    }

    val result = new Array[Byte](nElem * bytesPerElem)
    buf.flip()
    buf.get(result)
    result
  }

  /** Approximate Float32 → Float16 via truncation (matches ArrToStExpression). */
  private def floatToFloat16Truncate(f: Float): Short = {
    val bits   = java.lang.Float.floatToRawIntBits(f)
    val sign   = (bits >>> 31) & 0x1
    val exp32  = (bits >>> 23) & 0xff
    val mant32 = bits & 0x7fffff

    if (exp32 == 0xff) {
      ((sign << 15) | 0x7c00 | (if (mant32 != 0) 0x200 else 0)).toShort
    } else if (exp32 == 0) {
      (sign << 15).toShort
    } else {
      val exp16 = exp32 - 127 + 15
      if (exp16 >= 0x1f) ((sign << 15) | 0x7c00).toShort
      else if (exp16 <= 0) (sign << 15).toShort
      else ((sign << 15) | (exp16 << 10) | (mant32 >>> 13)).toShort
    }
  }

  /** Concatenate a sequence of byte arrays into a single array. */
  private def concatByteArrays(arrays: Array[Array[Byte]]): Array[Byte] = {
    val totalLen = arrays.map(_.length.toLong).sum
    val result   = new Array[Byte](totalLen.toInt)
    var pos      = 0
    arrays.foreach { a =>
      System.arraycopy(a, 0, result, pos, a.length)
      pos += a.length
    }
    result
  }

  // ---------------------------------------------------------------------------
  // Private helpers — shard lifecycle
  // ---------------------------------------------------------------------------

  private def writeBuffer(buf: ByteBuffer): Unit = {
    val bytes = new Array[Byte](buf.remaining())
    buf.get(bytes)
    currentShardStream.write(bytes)
  }

  private def openShardIfNeeded(): Unit =
    if (currentShardStream == null) {
      val fileName = f"part-${taskId}%05d-$taskUuid.safetensors"
      currentShardPath = new Path(outputPath, fileName).toString
      val fs = new Path(currentShardPath).getFileSystem(hadoopConf)
      currentShardStream = fs.create(new Path(currentShardPath))
      currentShardBytes = 0L
      currentShardSamples = 0
      openedShardPaths += currentShardPath
    }

  private def maybeSealShard(): Unit = {
    val thresholdBytes = options.targetShardSizeMb.toLong * 1024 * 1024
    if (currentShardBytes >= thresholdBytes) {
      sealCurrentShard()
    }
  }

  private def sealCurrentShard(): Unit = {
    if (currentShardStream == null) return
    closeShard()
  }

  private def closeShard(): Unit =
    if (currentShardStream != null) {
      currentShardStream.flush()
      currentShardStream.close()
      currentShardStream = null

      shards += ShardInfo(
        file = new Path(currentShardPath).getName,
        samplesCount = currentShardSamples,
        bytes = currentShardBytes
      )

      kvTensorBuffer.clear()
      currentShardTensorKeys.clear()
    }

}

/** Commit message returned by each DataWriter task to the driver's BatchWrite.commit(). Contains
  * per-shard statistics and optional tensor index entries for manifest and index assembly.
  */
final case class SafetensorsCommitMessage(
    shards: Seq[ShardInfo],
    indexEntries: Seq[TensorIndexEntry]
) extends WriterCommitMessage
