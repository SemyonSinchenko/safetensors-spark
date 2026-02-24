package io.github.semyonsinchenko.safetensors.benchmarks

import io.github.semyonsinchenko.safetensors.expressions.{ArrToStExpression, StToArrayExpression}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Literal
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType}
import org.apache.spark.unsafe.types.UTF8String
import org.openjdk.jmh.annotations._

import java.util.concurrent.TimeUnit

/** JMH benchmark for st_to_array deserialization expression.
  *
  * Tests performance of converting safetensors binary format back to Spark arrays.
  *
  * Benchmarks are parameterized by:
  *   - tensorSize: Number of float elements in the tensor (1K, 10K, 100K)
  *   - dtype: Source safetensors dtype (F32, F16)
  */
@BenchmarkMode(Array(Mode.Throughput))
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = Array("-Xmx2g"))
@Warmup(iterations = 3)
@Measurement(iterations = 5)
class StToArrayBenchmark {

  @Param(Array("1024", "10240", "102400")) // 1K, 10K, 100K elements
  var tensorSize: Int = _

  @Param(Array("F32", "F16"))
  var dtype: String = _

  private var inputRow: InternalRow           = _
  private var expression: StToArrayExpression = _

  @Setup(Level.Trial)
  def setup(): Unit = {
    // First create an array and convert to tensor using arr_to_st
    val floats     = Array.fill(tensorSize)(1.0f)
    val inputArray = new GenericArrayData(floats)
    val shape      = new GenericArrayData(Array(tensorSize))

    val arrToStExpr = ArrToStExpression(
      Literal(inputArray, ArrayType(FloatType)),
      Literal(shape, ArrayType(IntegerType)),
      Literal(UTF8String.fromString(dtype))
    )

    // Get the tensor struct row
    val tensorRow = arrToStExpr.eval(null).asInstanceOf[InternalRow]

    // Create the st_to_array expression
    expression = StToArrayExpression(Literal(tensorRow))
  }

  @Benchmark
  def convertTensorToArray(): Any =
    expression.eval(null)

}
