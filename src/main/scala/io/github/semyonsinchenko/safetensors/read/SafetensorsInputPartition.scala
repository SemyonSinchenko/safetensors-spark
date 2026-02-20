package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.connector.read.InputPartition

/**
 * One .safetensors file = one Spark InputPartition.
 *
 * @param filePath  Fully-qualified path to the .safetensors file
 *                  (hadoop-compatible URI, e.g. hdfs://.., s3a://.., file://..).
 */
final case class SafetensorsInputPartition(filePath: String) extends InputPartition
