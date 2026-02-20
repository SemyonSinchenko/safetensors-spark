package io.github.semyonsinchenko.safetensors.read

import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReader, PartitionReaderFactory}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/** Factory that creates a SafetensorsPartitionReader for each InputPartition.
  */
class SafetensorsPartitionReaderFactory(
    private val schema: StructType,
    private val options: CaseInsensitiveStringMap,
    private val pushedFilters: Array[Predicate],
    private val hadoopConf: Configuration
) extends PartitionReaderFactory {

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    partition match {
      case p: SafetensorsInputPartition =>
        new SafetensorsPartitionReader(p.filePath, schema, options, hadoopConf, p.requiredTensorKeys)
      case other =>
        throw new IllegalArgumentException(s"Unexpected partition type: ${other.getClass.getName}")
    }

}
