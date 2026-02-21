package io.github.semyonsinchenko.safetensors.read

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.SerializableConfiguration

/** Scan implementation for safetensors.
  *
  * Each .safetensors file becomes exactly one InputPartition (files are not splittable mid-file —
  * the full header must be read to locate any tensor).
  *
  * Path resolution: if a path is a directory, all direct-child .safetensors files are enumerated
  * (sorted by name for determinism). Files are never split mid-file.
  *
  * Pushed predicates are forwarded to each PartitionReader so that it can skip reading byte buffers
  * for tensor keys that do not satisfy a pushed tensor_key equality filter.
  */
class SafetensorsScan(
    private val schema: StructType,
    private val options: Map[String, String],
    private val paths: Seq[String],
    private val pushedFilters: Array[Predicate],
    private val hadoopConf: SerializableConfiguration
) extends Scan
    with Batch {

  // Unwrap once for local driver-side Hadoop FS operations
  private def conf: Configuration = hadoopConf.value

  override def readSchema(): StructType = schema

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] = {
    // Extract required tensor keys from the projected schema (column names)
    val requiredKeys = schema.fieldNames.toSet

    // Try to load the index for file-level predicate pushdown (§3.7)
    val indexMap: Map[String, Set[String]] = loadIndexMap()

    val partitions = paths.flatMap { rawPath =>
      val hadoopPath = new Path(rawPath)
      val fs         = FileSystem.get(hadoopPath.toUri, conf)

      if (!fs.exists(hadoopPath)) {
        Seq.empty[SafetensorsInputPartition]
      } else {
        val status = fs.getFileStatus(hadoopPath)
        if (!status.isDirectory) {
          // Direct file reference: always include (could be specifying a single file)
          if (rawPath.endsWith(".safetensors"))
            Seq(SafetensorsInputPartition(hadoopPath.toString, requiredKeys))
          else
            Seq.empty
        } else {
          // Directory: enumerate direct children that are .safetensors files,
          // sorted by name for deterministic partition ordering.
          // If index is available, filter to only files containing required keys.
          val allShardsInDir = fs
            .listStatus(hadoopPath)
            .filter(s => !s.isDirectory && s.getPath.getName.endsWith(".safetensors"))
            .sortBy(_.getPath.getName)
            .toSeq

          // Filter to files that contain at least one required tensor key (if index available)
          val filteredShards = if (indexMap.nonEmpty) {
            allShardsInDir.filter { status =>
              val fileName   = status.getPath.getName
              val keysInFile = indexMap.getOrElse(fileName, Set.empty)
              keysInFile.intersect(requiredKeys).nonEmpty
            }
          } else {
            allShardsInDir
          }

          filteredShards.map { s =>
            val fileName   = s.getPath.getName
            val keysInFile = indexMap.getOrElse(fileName, Set.empty)
            val partitionRequired =
              if (keysInFile.nonEmpty) keysInFile.intersect(requiredKeys) else requiredKeys
            SafetensorsInputPartition(s.getPath.toString, partitionRequired)
          }
        }
      }
    }
    partitions.toArray
  }

  /** Load the _tensor_index.parquet if it exists in any of the paths, building a map from fileName
    * → Set[tensorKey]. Returns empty map if no index is found or on read error.
    */
  private def loadIndexMap(): Map[String, Set[String]] = {
    try {
      val spark = org.apache.spark.sql.SparkSession.active
      for (rawPath <- paths) {
        val hadoopPath = new Path(rawPath)
        val fs         = FileSystem.get(hadoopPath.toUri, conf)
        if (fs.exists(hadoopPath)) {
          val status    = fs.getFileStatus(hadoopPath)
          val rootPath  = if (status.isDirectory) hadoopPath else hadoopPath.getParent
          val indexPath = new Path(rootPath, "_tensor_index.parquet")
          if (fs.exists(indexPath)) {
            // Read the index and build the map
            val indexDf = spark.read.parquet(indexPath.toString)
            val entries = indexDf
              .select("file_name", "tensor_key")
              .collect()

            val result: Map[String, Set[String]] = entries
              .groupBy(_.getString(0))
              .view
              .mapValues(rows => rows.map(_.getString(1)).toSet)
              .toMap

            return result
          }
        }
      }
    } catch {
      case scala.util.control.NonFatal(_) =>
      // Silently fall back to no index if reading fails
    }
    Map.empty
  }

  override def createReaderFactory(): PartitionReaderFactory =
    new SafetensorsPartitionReaderFactory(schema, options, pushedFilters, hadoopConf)

}
