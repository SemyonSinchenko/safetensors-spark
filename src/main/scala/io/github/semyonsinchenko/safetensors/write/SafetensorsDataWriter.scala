package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.core.{SafetensorsHeaderWriter, TensorSchema}
import io.github.semyonsinchenko.safetensors.manifest.ShardInfo
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write.{DataWriter, WriterCommitMessage}
import org.apache.spark.sql.types.{ArrayType, StructType}

import java.nio.ByteBuffer
import java.util.UUID

/**
 * DataWriter for safetensors output. Runs on each executor task.
 *
 * Output file naming: part-{taskId:05d}-{uuid}.safetensors
 * A new shard file is opened when estimated bytes exceed target_shard_size_mb.
 *
 * Write path memory model:
 *   - Tensor bytes from InternalRow (BinaryType) are wrapped with
 *     ByteBuffer.wrap() directly — no heap copy.
 *   - Write buffers for encoding numeric arrays use ByteBuffer.allocateDirect().
 *   - Hadoop FSDataOutputStream is used for all output path operations
 *     (supports HDFS, S3, GCS, Azure Blob).
 */
class SafetensorsDataWriter(
  private val partitionId: Int,
  private val taskId:      Long,
  private val schema:      StructType,
  private val options:     WriteOptions,
  private val outputPath:  String,
) extends DataWriter[InternalRow] {

  private val taskUuid = UUID.randomUUID().toString

  // Accumulated shard info for the commit message
  private val shards = scala.collection.mutable.ArrayBuffer.empty[ShardInfo]

  // Current shard state
  private var currentShardStream: org.apache.hadoop.fs.FSDataOutputStream = _
  private var currentShardPath:   String = _
  private var currentShardBytes:  Long   = 0L
  private var currentShardSamples: Int   = 0
  private var shardIndex:         Int    = 0

  // Pending rows in the current batch (batch_size mode)
  private val batchBuffer = scala.collection.mutable.ArrayBuffer.empty[InternalRow]

  private lazy val hadoopConf =
    SparkSession.active.sparkContext.hadoopConfiguration

  override def write(row: InternalRow): Unit = {
    options.namingStrategy match {
      case BatchSizeStrategy(batchSize) =>
        batchBuffer += row.copy()
        if (batchBuffer.size >= batchSize) {
          flushBatch()
        }
      case NameColStrategy(_) =>
        writeKVRow(row)
    }
  }

  override def commit(): WriterCommitMessage = {
    // Flush any remaining batch rows
    options.namingStrategy match {
      case _: BatchSizeStrategy if batchBuffer.nonEmpty => flushBatch()
      case _                                            =>
    }
    closeShard()
    SafetensorsCommitMessage(shards.toSeq)
  }

  override def abort(): Unit = {
    batchBuffer.clear()
    closeShard()
    // TODO: delete partial shard files
  }

  override def close(): Unit = closeShard()

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def flushBatch(): Unit = {
    if (batchBuffer.isEmpty) return

    val batchSize = batchBuffer.size
    openShardIfNeeded()

    // TODO: implement batch tensor stacking and writing
    // For each column: stack batchSize rows into one tensor, write to shard

    currentShardSamples += batchSize
    batchBuffer.clear()

    maybeSealShard()
  }

  private def writeKVRow(row: InternalRow): Unit = {
    openShardIfNeeded()

    // TODO: implement KV-mode row writing
    // - extract name from name_col
    // - handle duplicatesStrategy
    // - write tensor to current shard

    currentShardSamples += 1
    maybeSealShard()
  }

  private def openShardIfNeeded(): Unit = {
    if (currentShardStream == null) {
      val fileName = f"part-${taskId}%05d-$taskUuid-shard-${shardIndex}%04d.safetensors"
      currentShardPath = new Path(outputPath, fileName).toString
      val fs = new Path(currentShardPath).getFileSystem(hadoopConf)
      currentShardStream = fs.create(new Path(currentShardPath))
      currentShardBytes  = 0L
      currentShardSamples = 0
    }
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
    shardIndex += 1
  }

  private def closeShard(): Unit = {
    if (currentShardStream != null) {
      currentShardStream.flush()
      currentShardStream.close()
      currentShardStream = null

      shards += ShardInfo(
        file         = new Path(currentShardPath).getName,
        samplesCount = currentShardSamples,
        bytes        = currentShardBytes,
      )
    }
  }
}

/**
 * Commit message returned by each DataWriter task to the driver's
 * BatchWrite.commit(). Contains per-shard statistics for manifest assembly.
 */
final case class SafetensorsCommitMessage(
  shards: Seq[ShardInfo],
) extends WriterCommitMessage
