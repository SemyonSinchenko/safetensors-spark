package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.expressions.{ArrToStExpression, StToArrayExpression}
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.FunctionIdentifier

/** SparkSessionExtensions registration point.
  *
  * Enable via Spark config: spark.sql.extensions =
  * io.github.semyonsinchenko.safetensors.SafetensorsExtensions
  *
  * Registers:
  *   - arr_to_st(arrayCol, shape, dtype): ArrayType(FloatType) -> Tensor Struct
  *   - st_to_array(tensorCol): Tensor Struct -> ArrayType(FloatType)
  */
class SafetensorsExtensions extends (SparkSessionExtensions => Unit) {

  override def apply(extensions: SparkSessionExtensions): Unit = {
    val (arrName, arrInfo, arrBuilder) = ArrToStExpression.functionDescription
    extensions.injectFunction((FunctionIdentifier(arrName), arrInfo, arrBuilder))

    val (stName, stInfo, stBuilder) = StToArrayExpression.functionDescription
    extensions.injectFunction((FunctionIdentifier(stName), stInfo, stBuilder))
  }

}
