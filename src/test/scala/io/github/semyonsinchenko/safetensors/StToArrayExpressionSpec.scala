package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.TensorSchema
import io.github.semyonsinchenko.safetensors.expressions.StToArrayExpression

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, Literal}
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.{ByteBuffer, ByteOrder}

class StToArrayExpressionSpec extends AnyFlatSpec with Matchers {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /** Build an InternalRow representing a Tensor Struct. */
  private def makeTensorRow(data: Array[Byte], shape: Array[Int], dtype: String): InternalRow = {
    val row = new GenericInternalRow(3)
    row.update(0, data)
    row.update(1, new GenericArrayData(shape.map(_.asInstanceOf[AnyRef])))
    row.update(2, UTF8String.fromString(dtype))
    row
  }

  /** Evaluate st_to_array on a Tensor Struct row and return the decoded floats. */
  private def decode(data: Array[Byte], shape: Array[Int], dtype: String): Array[Float] = {
    val tensorRow = makeTensorRow(data, shape, dtype)
    val lit       = Literal.create(tensorRow, TensorSchema.schema)
    val expr      = StToArrayExpression(lit)
    val result    = expr.eval(InternalRow.empty)
    val arr       = result.asInstanceOf[org.apache.spark.sql.catalyst.util.ArrayData]
    Array.tabulate(arr.numElements())(arr.getFloat)
  }

  private def f32Bytes(floats: Float*): Array[Byte] = {
    val buf = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    floats.foreach(buf.putFloat)
    buf.array()
  }

  private def i32Bytes(ints: Int*): Array[Byte] = {
    val buf = ByteBuffer.allocate(ints.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    ints.foreach(buf.putInt)
    buf.array()
  }

  private def u64Bytes(longs: Long*): Array[Byte] = {
    val buf = ByteBuffer.allocate(longs.size * 8).order(ByteOrder.LITTLE_ENDIAN)
    longs.foreach(buf.putLong)
    buf.array()
  }

  private def bf16Bytes(floats: Float*): Array[Byte] = {
    val buf = ByteBuffer.allocate(floats.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    floats.foreach { f =>
      val bits = java.lang.Float.floatToRawIntBits(f)
      buf.putShort((bits >>> 16).toShort)
    }
    buf.array()
  }

  // ---------------------------------------------------------------------------
  // Type-check tests
  // ---------------------------------------------------------------------------

  "StToArrayExpression" should "pass type check for a valid Tensor Struct" in {
    val lit  = Literal.create(null, TensorSchema.schema)
    val expr = StToArrayExpression(lit)
    expr.checkInputDataTypes().isSuccess shouldBe true
  }

  it should "fail type check for a non-struct type" in {
    val lit  = Literal(UTF8String.fromString("not-a-struct"), StringType)
    val expr = StToArrayExpression(lit)
    expr.checkInputDataTypes().isSuccess shouldBe false
  }

  it should "fail type check for a StructType without the correct Tensor Struct fields" in {
    val badSchema = StructType(Seq(StructField("x", FloatType)))
    val lit       = Literal.create(null, badSchema)
    val expr      = StToArrayExpression(lit)
    expr.checkInputDataTypes().isSuccess shouldBe false
  }

  // ---------------------------------------------------------------------------
  // Decoding correctness tests
  // ---------------------------------------------------------------------------

  it should "decode F32 values correctly" in {
    val floats  = Array(1.0f, -2.5f, 3.14f)
    val decoded = decode(f32Bytes(1.0f, -2.5f, 3.14f), Array(3), "F32")
    decoded should have length 3
    decoded(0) shouldBe 1.0f +- 1e-6f
    decoded(1) shouldBe -2.5f +- 1e-6f
    decoded(2) shouldBe 3.14f +- 1e-4f
  }

  it should "decode I32 values as Float" in {
    val decoded = decode(i32Bytes(0, 42, -1), Array(3), "I32")
    decoded shouldBe Array(0.0f, 42.0f, -1.0f)
  }

  it should "decode U8 values correctly" in {
    val bytes   = Array(0.toByte, 128.toByte, 255.toByte)
    val decoded = decode(bytes, Array(3), "U8")
    decoded shouldBe Array(0.0f, 128.0f, 255.0f)
  }

  it should "decode BF16 values losslessly to Float32" in {
    // NOTE: BF16 is a special case outside the JSON schema — see §1.1
    // BF16→F32 is lossless: zero-extend top 16 bits into F32 mantissa
    val originalFloat = 1.5f
    val decoded       = decode(bf16Bytes(originalFloat), Array(1), "BF16")
    decoded should have length 1
    // BF16 truncation may differ from exact F32 representation by at most 1 ULP
    decoded(0) shouldBe originalFloat +- 0.01f
  }

  it should "decode U64 values correctly without string-parse overflow" in {
    // Test a value that fits in a signed Long
    val smallU64 = 42L
    val decoded1 = decode(u64Bytes(smallU64), Array(1), "U64")
    decoded1(0) shouldBe 42.0f +- 1.0f

    // Test the maximum signed Long value (high bit 0)
    val maxLong  = Long.MaxValue
    val decoded2 = decode(u64Bytes(maxLong), Array(1), "U64")
    // Should be approximately 9.22e18 (not negative, not a parse error)
    decoded2(0) should be > 0.0f

    // Test a value with the high bit set (true U64 > Long.MAX_VALUE)
    // In two's complement: -1L as U64 = 2^64 - 1 ≈ 1.8e19
    val maxU64bits = -1L
    val decoded3   = decode(u64Bytes(maxU64bits), Array(1), "U64")
    // Should be approximately 1.8e19, definitely > Long.MaxValue.toFloat
    decoded3(0) should be > Long.MaxValue.toFloat
  }

  // ---------------------------------------------------------------------------
  // Output type check
  // ---------------------------------------------------------------------------

  it should "return ArrayType(FloatType)" in {
    val lit  = Literal.create(null, TensorSchema.schema)
    val expr = StToArrayExpression(lit)
    expr.dataType shouldBe ArrayType(FloatType, containsNull = false)
  }

  // ---------------------------------------------------------------------------
  // Round-trip with ArrToStExpression
  // ---------------------------------------------------------------------------

  it should "round-trip F32 values through arr_to_st and st_to_array" in {
    import io.github.semyonsinchenko.safetensors.expressions.ArrToStExpression
    import org.apache.spark.sql.catalyst.util.GenericArrayData

    val floats    = Array(1.0f, 2.0f, 3.0f)
    val floatData = new GenericArrayData(floats.map(_.asInstanceOf[AnyRef]))
    val shapeData = new GenericArrayData(Array(3).map(_.asInstanceOf[AnyRef]))

    val arrLit   = Literal.create(floatData, ArrayType(FloatType))
    val shapeLit = Literal.create(shapeData, ArrayType(IntegerType))
    val dtypeLit = Literal(UTF8String.fromString("F32"), StringType)

    val arrToSt   = ArrToStExpression(arrLit, shapeLit, dtypeLit)
    val tensorRow = arrToSt.eval(InternalRow.empty).asInstanceOf[InternalRow]

    // Wrap the struct in a Literal for StToArrayExpression
    val structLit = Literal.create(tensorRow, TensorSchema.schema)
    val stToArr   = StToArrayExpression(structLit)
    val resultArr = stToArr
      .eval(InternalRow.empty)
      .asInstanceOf[org.apache.spark.sql.catalyst.util.ArrayData]

    val decoded = Array.tabulate(resultArr.numElements())(resultArr.getFloat)
    decoded shouldBe floats
  }

}
