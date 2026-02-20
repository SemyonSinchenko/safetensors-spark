package io.github.semyonsinchenko.safetensors.write

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.write.{DataWriter, DataWriterFactory}
import org.apache.spark.sql.types.StructType

/** Factory that creates one SafetensorsDataWriter per task. Runs on executors.
  */
class SafetensorsDataWriterFactory(
    private val schema: StructType,
    private val options: WriteOptions,
    private val outputPath: String
) extends DataWriterFactory {

  override def createWriter(partitionId: Int, taskId: Long): DataWriter[InternalRow] =
    new SafetensorsDataWriter(partitionId, taskId, schema, options, outputPath)

}
