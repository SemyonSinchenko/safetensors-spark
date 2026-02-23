package io.github.semyonsinchenko.safetensors.util

import org.apache.spark.sql.AnalysisException

/** Spark 4.0.x shim for AnalysisException construction.
  *
  * In Spark 4.0.x all AnalysisException constructors require a registered errorClass and
  * messageParameters: Map[String, String]. There is no free-form message constructor, and
  * custom error class names are not allowed — they must exist in Spark's internal
  * error-classes registry. We use the built-in "INTERNAL_ERROR" class, which accepts a
  * single "message" parameter, to carry our human-readable error text.
  */
object Errors {

  def analysisException(message: String): AnalysisException =
    new AnalysisException(
      errorClass = "INTERNAL_ERROR",
      messageParameters = Map("message" -> message)
    )

}
