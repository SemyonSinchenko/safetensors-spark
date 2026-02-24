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
  * Output file naming: part-{taskId:05d}-{shardIndex:04d}-{uuid}.safetensors (per spec §3.8). The
  * shard index is incremented per file produced by this task, ensuring unique filenames even when a
  * task produces multiple output files.
  *
  * ==Batch mode (batch_size)==
  *
  * Rows are accumulated in a buffer. When the buffer reaches batch_size, one complete, self-
  * contained safetensors file is written and immediately closed. Each output file contains exactly
  * batch_size stacked rows (one tensor per schema column). The leading tensor dimension is
  * batch_size; subsequent dimensions come from the shapes option or are inferred from the first
  * row.
  *
  * The tail of each partition (the remaining rows when partition size is not a multiple of
  * batch_size) is handled according to the tail_strategy option:
  *   - drop (default): the incomplete tail batch is discarded.
  *   - pad: the tail is zero-padded to reach exactly batch_size rows.
  *   - write: the tail is written as-is (smaller leading dimension).
  *
  * Note: each partition writes its batches independently. If the upstream DataFrame has skewed
  * partition sizes, some partitions will produce more output files than others. Repartition the
  * DataFrame before writing if balanced shard counts are required.
  *
  * ==KV mode (name_col)==
  *
  * Each row produces one tensor per non-key column. The tensor key is constructed as
  * {name_col_value}{kv_separator}{column_name}. Tensors are accumulated in a shard buffer and
  * flushed to a new file when the estimated byte count exceeds target_shard_size_mb.
  *
  * ==Write path memory model (§3.10)==
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

  // Per-task shard counter — incremented each time a new shard file is sealed.
  // Incorporated into the filename to prevent collisions when a task produces multiple files.
  private var shardIndex: Int = 0

  // Accumulated shard info for the commit message
  private val shards = scala.collection.mutable.ArrayBuffer.empty[ShardInfo]

  // Tensor index entries for the optional _tensor_index.parquet
  private val indexEntries = scala.collection.mutable.ArrayBuffer.empty[TensorIndexEntry]

  // Tracks every shard file path opened by this writer for abort cleanup
  private val openedShardPaths = scala.collection.mutable.ArrayBuffer.empty[String]

  // Pending rows in the current batch (batch_size mode)
  private val batchBuffer = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  // Accumulated tensors in current KV shard — (descriptor, raw bytes) pairs
  private val kvTensorBuffer =
    scala.collection.mutable.ArrayBuffer.empty[(TensorDescriptor, Array[Byte])]

  // Tracked tensor keys in current KV shard (for duplicate detection)
  private val currentShardTensorKeys = scala.collection.mutable.Set.empty[String]

  // KV shard size tracking
  private var kvShardBytes: Long  = 0L
  private var kvShardSamples: Int = 0

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
      case bs: BatchSizeStrategy =>
        handleTailBatch(bs.batchSize)
      case _: NameColStrategy =>
        if (kvTensorBuffer.nonEmpty) {
          flushKVShard()
        }
    }
    SafetensorsCommitMessage(shards.toSeq, indexEntries.toSeq)
  }

  override def abort(): Unit = {
    batchBuffer.clear()
    kvTensorBuffer.clear()
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

  override def close(): Unit = () // all streams are closed eagerly after each shard

  // ---------------------------------------------------------------------------
  // Private helpers — batch mode
  // ---------------------------------------------------------------------------

  /** Flush the current batchBuffer as one complete, self-contained safetensors file.
    *
    * Uses two passes over the buffer:
    *   - Pass 1: compute byte lengths per column (no encoding) to build descriptors and the header.
    *   - Pass 2: encode and stream each column's data row by row — no large concat allocation.
    *
    * Each call to flushBatch() opens a new shard file, writes a valid safetensors binary (8-byte
    * header-length prefix + JSON header + tensor data), and closes the stream immediately. The file
    * is therefore always a valid, standalone safetensors file.
    */
  private def flushBatch(): Unit = {
    if (batchBuffer.isEmpty) return

    val batchSize = batchBuffer.size

    // Resolve dtype and per-sample shape for each column (needed for both passes)
    val columnMeta: Seq[(String, Int, DataType, SafetensorsDtype, Seq[Int])] = columnsToWrite.map {
      case (colName, colIdx, colType) =>
        val dtype = options.dtype.getOrElse {
          colType match {
            case st: StructType if TensorSchema.isTensorStruct(st) =>
              SafetensorsDtype.fromStringUnsafe(
                batchBuffer.head.getStruct(colIdx, 3).getUTF8String(2).toString
              )
            case _ =>
              throw new IllegalStateException(
                s"dtype option is required for numeric array column '$colName'"
              )
          }
        }
        val perSampleShape = colType match {
          case st: StructType if TensorSchema.isTensorStruct(st) =>
            val shapeArr = batchBuffer.head.getStruct(colIdx, 3).getArray(1)
            (0 until shapeArr.numElements()).map(i => shapeArr.getInt(i))
          case _: ArrayType =>
            options.shapes.get(colName) match {
              case Some(s) => s
              case None    => Seq(batchBuffer.head.getArray(colIdx).numElements())
            }
          case other =>
            throw new IllegalStateException(
              s"Unexpected column type for '$colName': ${other.simpleString}"
            )
        }
        (colName, colIdx, colType, dtype, perSampleShape)
    }

    // Pass 1: compute byte lengths per column (no encoding) to build descriptors
    val descriptors: Seq[TensorDescriptor] = columnMeta.map {
      case (colName, colIdx, colType, dtype, perSampleShape) =>
        val totalBytes =
          batchBuffer.map(row => computeTensorByteLength(row, colIdx, colType, dtype)).sum
        TensorDescriptor(colName, dtype, Seq(batchSize) ++ perSampleShape, totalBytes)
    }

    val fileName = f"part-${taskId}%05d-${shardIndex}%04d-$taskUuid.safetensors"
    val filePath = new Path(outputPath, fileName).toString
    val fs       = new Path(filePath).getFileSystem(hadoopConf)
    val stream   = fs.create(new Path(filePath))
    openedShardPaths += filePath

    try {
      // Write header
      val headerBuf   = SafetensorsHeaderWriter.buildHeader(descriptors)
      val headerBytes = headerBuf.remaining()
      val headerArr   = new Array[Byte](headerBytes)
      headerBuf.get(headerArr)
      stream.write(headerArr)

      // Pass 2: encode and stream each column's data row by row — no concat allocation
      columnMeta.foreach { case (colName, colIdx, colType, dtype, _) =>
        batchBuffer.foreach { row =>
          val bytes = colType match {
            case st: StructType if TensorSchema.isTensorStruct(st) =>
              row.getStruct(colIdx, 3).getBinary(0)
            case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
              encodeNumericArray(row.getArray(colIdx), at.elementType, dtype)
            case other =>
              throw new IllegalStateException(
                s"Unexpected column type for '$colName': ${other.simpleString}"
              )
          }
          stream.write(bytes)
        }
      }

      // Collect index entries
      descriptors.foreach { desc =>
        indexEntries += TensorIndexEntry(
          tensorKey = desc.name,
          fileName = fileName,
          shape = desc.shape,
          dtype = desc.dtype.name
        )
      }

      val totalBytes = headerBytes.toLong + descriptors.map(_.byteLength).sum
      stream.flush()
      stream.close()

      shards += ShardInfo(shardPath = fileName, samplesCount = batchSize, bytes = totalBytes)
    } catch {
      case e: Exception =>
        try stream.close()
        catch { case _: Exception => () }
        throw e
    }

    shardIndex += 1
    batchBuffer.clear()
  }

  /** Handle the tail batch (remaining rows) according to tail_strategy. */
  private def handleTailBatch(batchSize: Int): Unit = {
    if (batchBuffer.isEmpty) return

    options.tailStrategy match {
      case DropTail =>
        batchBuffer.clear()

      case WriteAsIs =>
        flushBatch() // writes the partial batch as-is

      case PadWithZeros =>
        val remaining = batchSize - batchBuffer.size
        // Pad with zero-filled copies of the row structure
        val paddingRow = buildZeroPaddingRow()
        (0 until remaining).foreach(_ => batchBuffer += paddingRow)
        flushBatch()
    }
  }

  /** Build a zero-valued InternalRow with the same structure as the real rows.
    *
    * For Tensor Struct columns: uses the shape/dtype from the last real row but replaces data bytes
    * with zeros of the same length. For numeric array columns: uses an array of zeros.
    */
  private def buildZeroPaddingRow(): InternalRow = {
    import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
    import org.apache.spark.sql.catalyst.util.GenericArrayData
    import org.apache.spark.unsafe.types.UTF8String

    val fields = schema.fields.map { field =>
      val colType = field.dataType

      colType match {
        case st: StructType if TensorSchema.isTensorStruct(st) =>
          // Clone shape/dtype from the last buffered real row, replace data with zeros
          val lastRow      = batchBuffer.last
          val colIdx       = schema.fieldIndex(field.name)
          val struct       = lastRow.getStruct(colIdx, 3)
          val dataLen      = struct.getBinary(0).length
          val shapeArr     = struct.getArray(1)
          val zeroData     = new Array[Byte](dataLen)
          val paddedStruct = new GenericInternalRow(3)
          paddedStruct.update(0, zeroData)
          paddedStruct.update(1, shapeArr)
          paddedStruct.update(2, struct.getUTF8String(2))
          paddedStruct

        case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
          // Match element count from the last row; fill with zeros
          val lastRow = batchBuffer.last
          val colIdx  = schema.fieldIndex(field.name)
          val n       = lastRow.getArray(colIdx).numElements()
          val zeros = at.elementType match {
            case FloatType   => Array.fill[Any](n)(0.0f)
            case DoubleType  => Array.fill[Any](n)(0.0d)
            case IntegerType => Array.fill[Any](n)(0)
            case LongType    => Array.fill[Any](n)(0L)
            case ShortType   => Array.fill[Any](n)(0.toShort)
            case ByteType    => Array.fill[Any](n)(0.toByte)
            case _           => Array.fill[Any](n)(0)
          }
          new GenericArrayData(zeros)

        case _ =>
          null
      }
    }

    new GenericInternalRow(fields.asInstanceOf[Array[Any]])
  }

  /** Write tensors to a new, self-contained shard file and close the stream immediately.
    *
    * Used exclusively by KV mode where tensors are already fully materialised as Array[Byte].
    */
  private def writeShardFile(
      tensors: Seq[(TensorDescriptor, Array[Byte])],
      samplesInBatch: Int
  ): Unit = {
    val fileName = f"part-${taskId}%05d-${shardIndex}%04d-$taskUuid.safetensors"
    val filePath = new Path(outputPath, fileName).toString
    val fs       = new Path(filePath).getFileSystem(hadoopConf)
    val stream   = fs.create(new Path(filePath))
    openedShardPaths += filePath

    try {
      // Write safetensors header
      val descriptors = tensors.map(_._1)
      val headerBuf   = SafetensorsHeaderWriter.buildHeader(descriptors)
      val headerBytes = headerBuf.remaining()
      val headerArr   = new Array[Byte](headerBytes)
      headerBuf.get(headerArr)
      stream.write(headerArr)

      // Write tensor data in descriptor order
      val dataBytes = tensors.map { case (_, bytes) =>
        stream.write(bytes)
        bytes.length.toLong
      }.sum

      val totalBytes = headerBytes.toLong + dataBytes

      // Always collect index entries for manifest schema
      tensors.foreach { case (desc, _) =>
        indexEntries += TensorIndexEntry(
          tensorKey = desc.name,
          fileName = fileName,
          shape = desc.shape,
          dtype = desc.dtype.name
        )
      }

      stream.flush()
      stream.close()

      shards += ShardInfo(
        shardPath = fileName,
        samplesCount = samplesInBatch,
        bytes = totalBytes
      )
    } catch {
      case e: Exception =>
        try stream.close()
        catch { case _: Exception => () }
        throw e
    }

    shardIndex += 1
  }

  // ---------------------------------------------------------------------------
  // Private helpers — KV mode
  // ---------------------------------------------------------------------------

  private def writeKVRow(row: InternalRow): Unit = {
    val NameColStrategy(nameColName) = options.namingStrategy

    val nameColIdx   = schema.fieldIndex(nameColName)
    val nameColValue = row.getUTF8String(nameColIdx).toString

    for ((colName, colIdx, colType) <- columnsToWrite) {
      val tensorKey = s"$nameColValue${options.kvSeparator}$colName"

      val (tensorDescriptor, tensorBytes) = writeKVTensor(row, tensorKey, colName, colIdx, colType)

      if (currentShardTensorKeys.contains(tensorKey)) {
        options.duplicatesStrategy match {
          case FailOnDuplicate =>
            throw new IllegalStateException(
              s"Duplicate tensor key '$tensorKey' in name_col mode with duplicatesStrategy=fail. " +
                s"Either use duplicatesStrategy=lastWin or ensure unique keys."
            )
          case LastWinOnDuplicate =>
            kvTensorBuffer --= kvTensorBuffer.filter(_._1.name == tensorKey)
        }
      }

      currentShardTensorKeys += tensorKey
      kvTensorBuffer += ((tensorDescriptor, tensorBytes))
      kvShardBytes += tensorBytes.length.toLong + 200L // 200 B per-tensor header estimate
    }

    // Increment samples once per input row, not once per tensor column
    kvShardSamples += 1

    val thresholdBytes = options.targetShardSizeMb.toLong * 1024 * 1024
    if (kvShardBytes >= thresholdBytes) {
      flushKVShard()
    }
  }

  /** Flush accumulated KV tensors to a new, self-contained shard file. */
  private def flushKVShard(): Unit = {
    if (kvTensorBuffer.isEmpty) return

    val tensors        = kvTensorBuffer.toSeq
    val samplesInShard = kvShardSamples

    writeShardFile(tensors, samplesInShard)

    kvTensorBuffer.clear()
    currentShardTensorKeys.clear()
    kvShardBytes = 0L
    kvShardSamples = 0
  }

  /** Extract a single column from a row as a tensor descriptor + raw bytes. */
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
        val struct   = row.getStruct(colIdx, 3)
        val shapeArr = struct.getArray(1)
        val shape    = (0 until shapeArr.numElements()).map(i => shapeArr.getInt(i)).toSeq
        val data     = struct.getBinary(0)
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

  /** Compute the byte length a tensor column would produce for a given row, without encoding.
    *
    * Used in Pass 1 of flushBatch() to build TensorDescriptors before any data is written.
    */
  private def computeTensorByteLength(
      row: InternalRow,
      colIdx: Int,
      colType: DataType,
      dtype: SafetensorsDtype
  ): Long =
    colType match {
      case st: StructType if TensorSchema.isTensorStruct(st) =>
        row.getStruct(colIdx, 3).getBinary(0).length.toLong
      case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
        row.getArray(colIdx).numElements().toLong * SafetensorsDtype.bytesPerElement(dtype)
      case other =>
        throw new IllegalStateException(s"Unexpected column type: ${other.simpleString}")
    }

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

}

/** Commit message returned by each DataWriter task to the driver's BatchWrite.commit(). Contains
  * per-shard statistics and optional tensor index entries for manifest and index assembly.
  */
final case class SafetensorsCommitMessage(
    shards: Seq[ShardInfo],
    indexEntries: Seq[TensorIndexEntry]
) extends WriterCommitMessage
