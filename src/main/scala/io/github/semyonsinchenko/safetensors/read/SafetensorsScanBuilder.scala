package io.github.semyonsinchenko.safetensors.read

import org.apache.spark.sql.connector.read.{Scan, ScanBuilder, SupportsPushDownRequiredColumns, SupportsPushDownV2Filters}
import org.apache.spark.sql.connector.expressions.filter.Predicate
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * ScanBuilder for the safetensors DataSource V2.
 *
 * Implements:
 *   - SupportsPushDownRequiredColumns: skips tensor byte buffers for columns
 *     not in the projection (uses data_offsets from the header to seek past them).
 *   - SupportsPushDownV2Filters: pushes down tensor_key equality predicates
 *     (routing via _tensor_index.parquet if present; otherwise header scan).
 */
class SafetensorsScanBuilder(
  private var schema:  StructType,
  private val options: CaseInsensitiveStringMap,
  private val paths:   Seq[String],
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

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.schema = requiredSchema
  }

  // ---------------------------------------------------------------------------
  // SupportsPushDownV2Filters
  // ---------------------------------------------------------------------------

  override def pushPredicates(predicates: Array[Predicate]): Array[Predicate] = {
    // TODO: implement tensor_key equality pushdown
    // For now, all predicates are returned as post-scan filters
    this.pushedFilters   = Array.empty
    this.postScanFilters = predicates
    predicates
  }

  override def pushedPredicates(): Array[Predicate] = pushedFilters

  // ---------------------------------------------------------------------------
  // ScanBuilder
  // ---------------------------------------------------------------------------

  override def build(): Scan =
    new SafetensorsScan(schema, options, paths, pushedFilters)
}
