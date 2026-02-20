package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.read.{Batch, InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Scan implementation for safetensors.
 *
 * Each .safetensors file becomes exactly one InputPartition (files are not
 * splittable mid-file — the full header must be read to locate any tensor).
 */
class SafetensorsScan(
  private val schema:        StructType,
  private val options:       CaseInsensitiveStringMap,
  private val paths:         Seq[String],
  private val pushedFilters: Array[Predicate],
) extends Scan with Batch {

  override def readSchema(): StructType = schema

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] = {
    // TODO: resolve paths to individual .safetensors files via Hadoop FileSystem
    // and create one SafetensorsInputPartition per file
    Array.empty
  }

  override def createReaderFactory(): PartitionReaderFactory =
    new SafetensorsPartitionReaderFactory(schema, options)
}
