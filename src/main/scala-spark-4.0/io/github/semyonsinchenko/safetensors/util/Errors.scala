package io.github.semyonsinchenko.safetensors.util

import org.apache.spark.sql.AnalysisException

/** Spark 4.0.x shim for AnalysisException construction.
  *
  * In Spark 4.0.x all AnalysisException constructors require a bare (non-optional)
  * errorClass: String and messageParameters: Map[String, String]. There is no
  * free-form message constructor. We encode the human-readable message as the
  * value of a single "msg" parameter and use a fixed error class.
  */
object Errors {

  def analysisException(message: String): AnalysisException =
    new AnalysisException(
      errorClass = "SAFETENSORS_ERROR",
      messageParameters = Map("msg" -> message)
    )

}
