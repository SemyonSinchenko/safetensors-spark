package io.github.semyonsinchenko.safetensors.util

import org.apache.spark.sql.AnalysisException

/**
 * Helpers for constructing Spark AnalysisException in Spark 4.x.
 *
 * Spark 4 removed the single-String AnalysisException(message) constructor.
 * The closest compatible public constructor that accepts a free-form message is:
 *
 *   AnalysisException(
 *     message:           String,
 *     line:              Option[Int],
 *     startPosition:     Option[Int],
 *     cause:             Option[Throwable],
 *     errorClass:        Option[String],
 *     messageParameters: Map[String, String],
 *     context:           Array[QueryContext]
 *   )
 *
 * We use None/empty defaults for all optional fields except message.
 */
object Errors {

  def analysisException(message: String): AnalysisException =
    new AnalysisException(
      message           = message,
      line              = None,
      startPosition     = None,
      cause             = None,
      errorClass        = Some("SAFETENSORS_ERROR"),
      messageParameters = Map.empty,
      context           = Array.empty,
    )
}
