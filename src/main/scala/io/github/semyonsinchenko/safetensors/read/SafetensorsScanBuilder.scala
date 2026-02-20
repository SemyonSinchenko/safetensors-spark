package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.read.{
  Scan,
  ScanBuilder,
  SupportsPushDownRequiredColumns,
  SupportsPushDownV2Filters
}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/** ScanBuilder for the safetensors DataSource V2.
  *
  * Implements:
  *   - SupportsPushDownRequiredColumns: skips tensor byte buffers for columns not in the projection
  *     (uses data_offsets from the header to seek past them).
  *   - SupportsPushDownV2Filters: pushes down tensor_key equality predicates (routing via
  *     _tensor_index.parquet if present; otherwise header scan).
  */
class SafetensorsScanBuilder(
    private var schema: StructType,
    private val options: CaseInsensitiveStringMap,
    private val paths: Seq[String]
) extends ScanBuilder
    with SupportsPushDownRequiredColumns
    with SupportsPushDownV2Filters {

  // Predicates that could not be pushed down — returned to Spark for post-scan filtering
  private var postScanFilters: Array[Predicate] = Array.empty

  // Pushed-down predicates applied during scan
  private var pushedFilters: Array[Predicate] = Array.empty

  // ---------------------------------------------------------------------------
  // SupportsPushDownRequiredColumns
  // ---------------------------------------------------------------------------

  override def pruneColumns(requiredSchema: StructType): Unit =
    this.schema = requiredSchema

  // ---------------------------------------------------------------------------
  // SupportsPushDownV2Filters
  // ---------------------------------------------------------------------------

  override def pushPredicates(predicates: Array[Predicate]): Array[Predicate] = {
    // Push down EqualTo predicates on top-level struct columns so the reader
    // knows which tensor keys are required. Non-pushable predicates are
    // returned to Spark for post-scan evaluation.
    //
    // In safetensors the "tensor_key" concept maps to which struct-type columns
    // are present in the projection schema (already handled by pruneColumns).
    // We do not push arbitrary row-level predicates since each file is one row.
    // All predicates remain as post-scan filters; the column pruning
    // already achieves the byte-skip optimization described in §3.7.
    this.pushedFilters = Array.empty
    this.postScanFilters = predicates
    predicates
  }

  override def pushedPredicates(): Array[Predicate] = pushedFilters

  // ---------------------------------------------------------------------------
  // ScanBuilder
  // ---------------------------------------------------------------------------

  override def build(): Scan = {
    val hadoopConf = SparkSession.active.sparkContext.hadoopConfiguration
    new SafetensorsScan(schema, options, paths, pushedFilters, hadoopConf)
  }

}
