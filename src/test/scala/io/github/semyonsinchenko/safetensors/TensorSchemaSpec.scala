package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.TensorSchema
import org.apache.spark.sql.types._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TensorSchemaSpec extends AnyFlatSpec with Matchers {

  "TensorSchema.isTensorStruct" should "recognise the canonical Tensor Struct" in {
    TensorSchema.isTensorStruct(TensorSchema.schema) shouldBe true
  }

  it should "reject a struct with wrong field names" in {
    val wrong = StructType(Seq(
      StructField("bytes", BinaryType, nullable = false),
      StructField("shape", ArrayType(IntegerType, false), nullable = false),
      StructField("dtype", StringType, nullable = false),
    ))
    TensorSchema.isTensorStruct(wrong) shouldBe false
  }

  it should "reject a struct with wrong field types" in {
    val wrong = StructType(Seq(
      StructField("data",  StringType, nullable = false),
      StructField("shape", ArrayType(IntegerType, false), nullable = false),
      StructField("dtype", StringType, nullable = false),
    ))
    TensorSchema.isTensorStruct(wrong) shouldBe false
  }

  "TensorSchema.isNumericArrayType" should "accept all numeric element types" in {
    Seq(FloatType, DoubleType, IntegerType, LongType, ShortType, ByteType).foreach { t =>
      TensorSchema.isNumericArrayType(ArrayType(t)) shouldBe true
    }
  }

  it should "reject non-numeric element types" in {
    TensorSchema.isNumericArrayType(ArrayType(StringType))  shouldBe false
    TensorSchema.isNumericArrayType(ArrayType(BooleanType)) shouldBe false
    TensorSchema.isNumericArrayType(StringType)             shouldBe false
  }
}
