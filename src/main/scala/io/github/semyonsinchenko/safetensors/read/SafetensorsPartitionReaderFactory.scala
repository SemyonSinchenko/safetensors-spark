package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReader, PartitionReaderFactory}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Factory that creates a SafetensorsPartitionReader for each InputPartition.
 */
class SafetensorsPartitionReaderFactory(
  private val schema:  StructType,
  private val options: CaseInsensitiveStringMap,
) extends PartitionReaderFactory {

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] = {
    partition match {
      case p: SafetensorsInputPartition =>
        new SafetensorsPartitionReader(p.filePath, schema, options)
      case other =>
        throw new IllegalArgumentException(
          s"Unexpected partition type: ${other.getClass.getName}")
    }
  }
}
