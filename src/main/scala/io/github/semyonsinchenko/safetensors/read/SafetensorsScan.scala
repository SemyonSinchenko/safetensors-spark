package io.github.semyonsinchenko.safetensors.read

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

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
    private val options: CaseInsensitiveStringMap,
    private val paths: Seq[String],
    private val pushedFilters: Array[Predicate],
    private val hadoopConf: Configuration
) extends Scan
    with Batch {

  override def readSchema(): StructType = schema

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] = {
    val partitions = paths.flatMap { rawPath =>
      val hadoopPath = new Path(rawPath)
      val fs         = FileSystem.get(hadoopPath.toUri, hadoopConf)

      if (!fs.exists(hadoopPath)) {
        Seq.empty[SafetensorsInputPartition]
      } else {
        val status = fs.getFileStatus(hadoopPath)
        if (!status.isDirectory) {
          // Direct file reference
          if (rawPath.endsWith(".safetensors"))
            Seq(SafetensorsInputPartition(hadoopPath.toString))
          else
            Seq.empty
        } else {
          // Directory: enumerate direct children that are .safetensors files,
          // sorted by name for deterministic partition ordering.
          fs.listStatus(hadoopPath)
            .filter(s => !s.isDirectory && s.getPath.getName.endsWith(".safetensors"))
            .sortBy(_.getPath.getName)
            .map(s => SafetensorsInputPartition(s.getPath.toString))
            .toSeq
        }
      }
    }
    partitions.toArray
  }

  override def createReaderFactory(): PartitionReaderFactory =
    new SafetensorsPartitionReaderFactory(schema, options, pushedFilters, hadoopConf)

}
