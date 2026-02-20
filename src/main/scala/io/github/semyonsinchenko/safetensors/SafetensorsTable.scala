package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.read.SafetensorsScanBuilder
import io.github.semyonsinchenko.safetensors.write.SafetensorsWriteBuilder
import org.apache.spark.sql.connector.catalog.{SupportsRead, SupportsWrite, Table, TableCapability}
import org.apache.spark.sql.connector.read.ScanBuilder
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util
import scala.jdk.CollectionConverters._

/**
 * Represents a safetensors "table" — a directory (or single file) containing
 * one or more .safetensors shard files.
 *
 * One .safetensors file = one Spark InputPartition (files are not splittable).
 */
class SafetensorsTable(
  private val tableSchema: StructType,
  private val options:     CaseInsensitiveStringMap,
  private val paths:       Seq[String],
) extends Table with SupportsRead with SupportsWrite {

  override def name(): String = s"safetensors(${paths.mkString(",")})"

  override def schema(): StructType = tableSchema

  override def capabilities(): util.Set[TableCapability] =
    Set(
      TableCapability.BATCH_READ,
      TableCapability.BATCH_WRITE,
      TableCapability.TRUNCATE,
    ).asJava

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new SafetensorsScanBuilder(tableSchema, this.options, paths)

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder =
    new SafetensorsWriteBuilder(info, options, paths)
}
