package io.github.semyonsinchenko.safetensors.benchmarks

import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.openjdk.jmh.annotations._

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.concurrent.TimeUnit
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** JMH benchmark comparing safetensors serialization vs naive Java serialization.
  *
  * Baseline comparison for measuring overhead and efficiency.
  *
  * Tests:
  *   - Safetensors-style direct byte buffer encoding
  *   - Naive Java serialization (ObjectOutputStream)
  */
@BenchmarkMode(Array(Mode.Throughput, Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Fork(value = 1, jvmArgs = Array("-Xmx2g"))
@Warmup(iterations = 3)
@Measurement(iterations = 5)
class BaselineComparisonBenchmark {

  @Param(Array("1024", "10240", "102400")) // 1K, 10K, 100K elements
  var arraySize: Int = _

  private var inputArray: GenericArrayData = _
  private var inputFloats: Array[Float] = _

  @Setup(Level.Trial)
  def setup(): Unit = {
    inputFloats = Array.fill(arraySize)(1.0f)
    inputArray = new GenericArrayData(inputFloats)
  }

  /** Safetensors-style encoding: direct ByteBuffer with little-endian floats. */
  @Benchmark
  def safetensorsStyleEncoding(): Array[Byte] = {
    val buf = ByteBuffer.allocate(arraySize * 4)
    buf.order(ByteOrder.LITTLE_ENDIAN)
    var i = 0
    while (i < arraySize) {
      buf.putFloat(inputFloats(i))
      i += 1
    }
    buf.array()
  }

  /** Naive Java serialization using ObjectOutputStream. */
  @Benchmark
  def naiveJavaSerialization(): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    val oos = new ObjectOutputStream(baos)
    oos.writeObject(inputFloats)
    oos.flush()
    baos.toByteArray()
  }

  /** Measure deserialization overhead for safetensors style. */
  @Benchmark
  def safetensorsStyleDecoding(): Array[Float] = {
    val data = safetensorsStyleEncoding()
    val buf = ByteBuffer.wrap(data)
    buf.order(ByteOrder.LITTLE_ENDIAN)
    val result = new Array[Float](arraySize)
    var i = 0
    while (i < arraySize) {
      result(i) = buf.getFloat()
      i += 1
    }
    result
  }

  /** Measure deserialization overhead for Java serialization. */
  @Benchmark
  def naiveJavaDeserialization(): Array[Float] = {
    val data = naiveJavaSerialization()
    val bais = new ByteArrayInputStream(data)
    val ois = new ObjectInputStream(bais)
    ois.readObject().asInstanceOf[Array[Float]]
  }

}
