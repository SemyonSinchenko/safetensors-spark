package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.manifest.{DatasetManifest, ShardInfo}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.write.{BatchWrite, DataWriterFactory, PhysicalWriteInfo, WriterCommitMessage}
import org.apache.spark.sql.types.StructType

import java.io.{OutputStreamWriter, PrintWriter}
import scala.util.control.NonFatal

/**
 * BatchWrite implementation. Runs on the driver.
 *
 * Coordinates commit/abort and assembles dataset_manifest.json from the
 * per-task WriterCommitMessages returned by each SafetensorsDataWriter.
 */
class SafetensorsBatchWrite(
  private val schema:   StructType,
  private val options:  WriteOptions,
  private val paths:    Seq[String],
) extends BatchWrite {

  private val outputPath = paths.headOption
    .getOrElse(throw new IllegalArgumentException("No output path specified"))

  override def createBatchWriterFactory(info: PhysicalWriteInfo): DataWriterFactory =
    new SafetensorsDataWriterFactory(schema, options, outputPath)

  override def commit(messages: Array[WriterCommitMessage]): Unit = {
    val commitMsgs = messages.collect { case m: SafetensorsCommitMessage => m }

    // Aggregate shard stats from all tasks
    val shards = commitMsgs.flatMap(_.shards).sortBy(_.file)

    val totalSamples = shards.map(_.samplesCount.toLong).sum
    val totalBytes   = shards.map(_.bytes).sum

    val manifest = DatasetManifest(
      formatVersion      = "1.0",
      safetensorsVersion = "1.0",
      totalSamples       = totalSamples,
      totalBytes         = totalBytes,
      shards             = shards.toSeq,
    )

    writeManifest(manifest)

    if (options.generateIndex) {
      writeTensorIndex(commitMsgs)
    }
  }

  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    // TODO: clean up partial output files written by tasks
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def writeManifest(manifest: DatasetManifest): Unit = {
    import com.fasterxml.jackson.databind.ObjectMapper
    import com.fasterxml.jackson.module.scala.DefaultScalaModule

    val spark    = SparkSession.active
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val outPath  = new Path(outputPath, "dataset_manifest.json")
    val fs       = outPath.getFileSystem(hadoopConf)

    val mapper   = new ObjectMapper().registerModule(DefaultScalaModule)
    val json     = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(manifest)

    val out = fs.create(outPath, true /* overwrite */)
    try {
      out.write(json.getBytes("UTF-8"))
    } finally {
      out.close()
    }
  }

  private def writeTensorIndex(messages: Array[SafetensorsCommitMessage]): Unit = {
    // TODO: collect TensorIndexEntry records from messages and write
    // _tensor_index.parquet via SparkSession.createDataFrame
  }
}
