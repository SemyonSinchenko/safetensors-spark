package io.github.semyonsinchenko.safetensors.util

import org.apache.spark.sql.AnalysisException

/** Spark 4.1.x shim for AnalysisException construction.
  *
  * In Spark 4.1.x the AnalysisException constructor accepts an optional errorClass
  * and a free-form message string, matching the original implementation.
  */
object Errors {

  def analysisException(message: String): AnalysisException =
    new AnalysisException(
      message = message,
      line = None,
      startPosition = None,
      cause = None,
      errorClass = Some("SAFETENSORS_ERROR"),
      messageParameters = Map.empty,
      context = Array.empty
    )

}
