package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.connector.read.InputPartition

/** One .safetensors file = one Spark InputPartition.
  *
  * @param filePath
  *   Fully-qualified path to the .safetensors file (hadoop-compatible URI, e.g. hdfs://..,
  *   s3a://.., file://..).
  * @param requiredTensorKeys
  *   Set of tensor keys that must be present in this file for it to be read. Empty set means all
  *   keys are required (no pushdown filtering). Used by predicate pushdown to skip files that don't
  *   contain any of the required tensor columns.
  */
final case class SafetensorsInputPartition(
    filePath: String,
    requiredTensorKeys: Set[String] = Set.empty
) extends InputPartition
