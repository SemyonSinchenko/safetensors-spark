package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.expressions.ArrToStExpression

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, Literal}
import org.apache.spark.sql.catalyst.util.{ArrayData, GenericArrayData}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.{ByteBuffer, ByteOrder}

class ArrToStExpressionSpec extends AnyFlatSpec with Matchers {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private def makeExpr(dtype: String): ArrToStExpression = {
    val arrayLit = Literal.create(null, ArrayType(FloatType))
    val shapeLit = Literal.create(null, ArrayType(IntegerType))
    val dtypeLit = Literal(UTF8String.fromString(dtype), StringType)
    ArrToStExpression(arrayLit, shapeLit, dtypeLit)
  }

  /** Evaluate arr_to_st with a literal float array and shape. */
  private def encode(
      floats: Array[Float],
      shape: Array[Int],
      dtype: String
  ): InternalRow = {
    val arrayData = new GenericArrayData(floats.map(f => f.asInstanceOf[AnyRef]))
    val shapeData = new GenericArrayData(shape.map(i => i.asInstanceOf[AnyRef]))
    val arrayLit  = Literal.create(arrayData, ArrayType(FloatType))
    val shapeLit  = Literal.create(shapeData, ArrayType(IntegerType))
    val dtypeLit  = Literal(UTF8String.fromString(dtype), StringType)
    val expr      = ArrToStExpression(arrayLit, shapeLit, dtypeLit)
    expr.eval(InternalRow.empty).asInstanceOf[InternalRow]
  }

  private def readF32LE(bytes: Array[Byte]): Array[Float] = {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.fill(bytes.length / 4)(buf.getFloat())
  }

  private def readF64LE(bytes: Array[Byte]): Array[Double] = {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.fill(bytes.length / 8)(buf.getDouble())
  }

  private def readI32LE(bytes: Array[Byte]): Array[Int] = {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    Array.fill(bytes.length / 4)(buf.getInt())
  }

  // ---------------------------------------------------------------------------
  // Type-check tests
  // ---------------------------------------------------------------------------

  "ArrToStExpression" should "pass type check for valid inputs" in {
    val expr = makeExpr("F32")
    expr.checkInputDataTypes().isSuccess shouldBe true
  }

  it should "fail type check when arrayCol is not ArrayType(FloatType)" in {
    val badArray = Literal.create(null, ArrayType(IntegerType))
    val shapeLit = Literal.create(null, ArrayType(IntegerType))
    val dtypeLit = Literal(UTF8String.fromString("F32"), StringType)
    val expr     = ArrToStExpression(badArray, shapeLit, dtypeLit)
    expr.checkInputDataTypes().isSuccess shouldBe false
  }

  it should "fail type check when shape is not ArrayType(IntegerType)" in {
    val arrayLit = Literal.create(null, ArrayType(FloatType))
    val badShape = Literal.create(null, ArrayType(FloatType))
    val dtypeLit = Literal(UTF8String.fromString("F32"), StringType)
    val expr     = ArrToStExpression(arrayLit, badShape, dtypeLit)
    expr.checkInputDataTypes().isSuccess shouldBe false
  }

  it should "fail type check when dtype is not StringType" in {
    val arrayLit = Literal.create(null, ArrayType(FloatType))
    val shapeLit = Literal.create(null, ArrayType(IntegerType))
    val badDtype = Literal(42, IntegerType)
    val expr     = ArrToStExpression(arrayLit, shapeLit, badDtype)
    expr.checkInputDataTypes().isSuccess shouldBe false
  }

  // ---------------------------------------------------------------------------
  // Encoding correctness tests
  // ---------------------------------------------------------------------------

  it should "encode F32 values as little-endian IEEE 754 bytes" in {
    val floats    = Array(1.0f, 2.0f, 3.0f)
    val row       = encode(floats, Array(3), "F32")
    val dataBytes = row.getBinary(0)
    val decoded   = readF32LE(dataBytes)
    decoded shouldBe floats
  }

  it should "encode F64 values as little-endian doubles" in {
    val floats    = Array(1.5f, -2.5f)
    val row       = encode(floats, Array(2), "F64")
    val dataBytes = row.getBinary(0)
    val decoded   = readF64LE(dataBytes)
    decoded.map(_.toFloat) shouldBe floats
  }

  it should "encode I32 values correctly" in {
    val floats    = Array(42.0f, -1.0f, 0.0f)
    val row       = encode(floats, Array(3), "I32")
    val dataBytes = row.getBinary(0)
    val decoded   = readI32LE(dataBytes)
    decoded shouldBe Array(42, -1, 0)
  }

  it should "encode U8 values correctly" in {
    val floats    = Array(0.0f, 128.0f, 255.0f)
    val row       = encode(floats, Array(3), "U8")
    val dataBytes = row.getBinary(0)
    dataBytes.map(_ & 0xff) shouldBe Array(0, 128, 255)
  }

  it should "produce correct byte length for each dtype" in {
    val floats = Array(1.0f, 2.0f, 3.0f, 4.0f) // 4 elements
    val dtypeByteCounts = Map(
      "F32"  -> 4 * 4,
      "F64"  -> 4 * 8,
      "I32"  -> 4 * 4,
      "I16"  -> 4 * 2,
      "U16"  -> 4 * 2,
      "U8"   -> 4 * 1,
      "I8"   -> 4 * 1,
      "I64"  -> 4 * 8,
      "U64"  -> 4 * 8,
      "BF16" -> 4 * 2,
      "F16"  -> 4 * 2
    )
    dtypeByteCounts.foreach { case (dtype, expectedBytes) =>
      val row       = encode(floats, Array(4), dtype)
      val dataBytes = row.getBinary(0)
      withClue(s"dtype=$dtype: ") {
        dataBytes.length shouldBe expectedBytes
      }
    }
  }

  it should "preserve BF16 encoding as top-16-bit truncation of Float32" in {
    // NOTE: BF16 is not in the JSON schema regex — see §1.1
    val f     = 1.5f
    val row   = encode(Array(f), Array(1), "BF16")
    val bytes = row.getBinary(0)
    bytes should have length 2

    // BF16 = top 16 bits of float32 bit pattern
    val f32bits  = java.lang.Float.floatToRawIntBits(f)
    val expected = (f32bits >>> 16).toShort
    val actual   = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort()
    actual shouldBe expected
  }

  it should "pass through the shape array unchanged" in {
    val row       = encode(Array(1.0f, 2.0f), Array(2, 1), "F32")
    val shapeData = row.getArray(1)
    shapeData.numElements() shouldBe 2
    shapeData.getInt(0) shouldBe 2
    shapeData.getInt(1) shouldBe 1
  }

  it should "store the dtype string in the struct" in {
    val row = encode(Array(1.0f), Array(1), "F32")
    row.getUTF8String(2).toString shouldBe "F32"
  }

}
