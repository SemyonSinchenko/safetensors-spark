package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.manifest.{DatasetManifest, ShardInfo, TensorIndexEntry}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.connector.write.{
  BatchWrite,
  DataWriterFactory,
  PhysicalWriteInfo,
  WriterCommitMessage
}
import org.apache.spark.sql.types.{ArrayType, IntegerType, StringType, StructField, StructType}

import scala.util.control.NonFatal

/** BatchWrite implementation. Runs on the driver.
  *
  * Coordinates commit/abort and assembles dataset_manifest.json from the per-task
  * WriterCommitMessages returned by each SafetensorsDataWriter. Optionally writes
  * _tensor_index.parquet when generate_index=true.
  */
class SafetensorsBatchWrite(
    private val schema: StructType,
    private val options: WriteOptions,
    private val paths: Seq[String]
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
      formatVersion = "1.0",
      safetensorsVersion = "1.0",
      totalSamples = totalSamples,
      totalBytes = totalBytes,
      shards = shards.toSeq
    )

    writeManifest(manifest)

    if (options.generateIndex) {
      val entries = commitMsgs.flatMap(_.indexEntries).toSeq
      writeTensorIndex(entries)
    }
  }

  override def abort(messages: Array[WriterCommitMessage]): Unit = {
    // Delete all partial shard files reported by tasks that did commit
    val commitMsgs = messages.collect { case m: SafetensorsCommitMessage => m }
    val spark      = SparkSession.active
    val hadoopConf = spark.sparkContext.hadoopConfiguration

    commitMsgs.flatMap(_.shards).foreach { shard =>
      try {
        val shardPath = new Path(outputPath, shard.file)
        val fs        = FileSystem.get(shardPath.toUri, hadoopConf)
        fs.delete(shardPath, false)
      } catch {
        case NonFatal(_) => // best effort
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def writeManifest(manifest: DatasetManifest): Unit = {
    import com.fasterxml.jackson.databind.ObjectMapper
    import com.fasterxml.jackson.module.scala.DefaultScalaModule

    val spark      = SparkSession.active
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val outPath    = new Path(outputPath, "dataset_manifest.json")
    val fs         = outPath.getFileSystem(hadoopConf)

    val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
    val json   = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(manifest)

    val out = fs.create(outPath, true /* overwrite */ )
    try
      out.write(json.getBytes("UTF-8"))
    finally
      out.close()
  }

  /** Write _tensor_index.parquet at the output root.
    *
    * Schema (per §3.6): tensor_key : StringType file_name : StringType shape :
    * ArrayType(IntegerType) dtype : StringType
    */
  private def writeTensorIndex(entries: Seq[TensorIndexEntry]): Unit = {
    if (entries.isEmpty) return

    val spark = SparkSession.active

    val indexSchema = StructType(
      Seq(
        StructField("tensor_key", StringType, nullable = false),
        StructField("file_name", StringType, nullable = false),
        StructField("shape", ArrayType(IntegerType), nullable = false),
        StructField("dtype", StringType, nullable = false)
      )
    )

    val rows = entries.map { e =>
      Row(e.tensorKey, e.fileName, e.shape, e.dtype)
    }

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      indexSchema
    )

    val indexPath = new Path(outputPath, "_tensor_index.parquet").toString
    df.write.mode("overwrite").parquet(indexPath)
  }

}
