package io.github.semyonsinchenko.safetensors.benchmarks

import io.github.semyonsinchenko.safetensors.core.SafetensorsDtype
import io.github.semyonsinchenko.safetensors.expressions.ArrToStExpression
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType}
import org.apache.spark.unsafe.types.UTF8String
import org.openjdk.jmh.annotations._

import java.util.concurrent.TimeUnit

/** JMH benchmark for array_to_st conversion expression.
  *
  * Tests performance with varying tensor sizes: 1K, 10K, 100K elements.
  *
  * Benchmarks are parameterized by:
  *   - arraySize: Number of float elements in the input array (1K, 10K, 100K)
  *   - dtype: Target safetensors dtype (F32, F16, BF16)
  */
@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = Array("-Xmx2g"))
@Warmup(iterations = 3)
@Measurement(iterations = 5)
class ArrToStBenchmark {

  @Param(Array("1024", "10240", "102400")) // 1K, 10K, 100K elements
  var arraySize: Int = _

  @Param(Array("F32", "F16", "BF16"))
  var dtype: String = _

  private var inputArray: GenericArrayData  = _
  private var expression: ArrToStExpression = _

  @Setup(Level.Trial)
  def setup(): Unit = {
    // Create input array with sample data
    val floats = Array.fill(arraySize)(1.0f)
    inputArray = new GenericArrayData(floats)

    // Create the expression
    val shape = new GenericArrayData(Array(arraySize))
    expression = ArrToStExpression(
      Literal(inputArray, ArrayType(FloatType)),
      Literal(shape, ArrayType(org.apache.spark.sql.types.IntegerType)),
      Literal(UTF8String.fromString(dtype))
    )
  }

  @Benchmark
  def convertArrayToTensor(): Any =
    // Evaluate the expression (null input for eval since it's not bound to row)
    // For proper benchmark, we need a mock InternalRow context
    expression.eval(null)

}
