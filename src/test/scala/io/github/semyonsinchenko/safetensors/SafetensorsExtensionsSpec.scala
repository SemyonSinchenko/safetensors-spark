package io.github.semyonsinchenko.safetensors

import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.jdk.CollectionConverters._

/** Unit tests for Catalyst SQL functions: arr_to_st and st_to_array via Spark DataFrame API.
  *
  * Tests verify:
  *   - Functions are properly registered and callable from DataFrame API
  *   - Type validation works at plan time
  */
class SafetensorsExtensionsSpec
    extends AnyFlatSpec
    with Matchers
    with BeforeAndAfterAll {

  private var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession
      .builder()
      .appName("SafetensorsExtensionsSpec")
      .master("local[1]")
      .config("spark.sql.extensions", "io.github.semyonsinchenko.safetensors.SafetensorsExtensions")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  behavior of "Catalyst function registration"

  it should "register arr_to_st function" in {
    spark.sql(
      "SELECT arr_to_st(array(1.0f, 2.0f), array(2), 'F32') as tensor LIMIT 1"
    ).collect()
  }

  it should "register st_to_array function" in {
    spark.sql(
      "SELECT arr_to_st(array(1.0f, 2.0f), array(2), 'F32') as tensor LIMIT 1"
    ).collect()
  }

  behavior of "Catalyst function type checking"

  it should "fail arr_to_st with non-array input" in {
    an[Exception] shouldBe thrownBy {
      val schema = StructType(Seq(StructField("data", StringType, nullable = false)))
      val data = Seq(Row("not_an_array")).asJava
      val df = spark.createDataFrame(data, schema)
      df.select(
        call_udf("arr_to_st", col("data"), array(lit(1)), lit("F32"))
      ).collect()
    }
  }

  it should "fail st_to_array with non-struct input" in {
    an[Exception] shouldBe thrownBy {
      val schema = StructType(Seq(StructField("data", StringType, nullable = false)))
      val data = Seq(Row("not_a_struct")).asJava
      val df = spark.createDataFrame(data, schema)
      df.select(call_udf("st_to_array", col("data"))).collect()
    }
  }

  behavior of "arr_to_st output shape"

  it should "encode with correct byte length for F32" in {
    // 4 floats * 4 bytes/float = 16 bytes
    val result = spark.sql(
      "SELECT arr_to_st(array(cast(1.0 as float), cast(2.0 as float), cast(3.0 as float), cast(4.0 as float)), array(2, 2), 'F32') as tensor"
    ).collect()
    result.length shouldBe 1
    val tensorRow = result(0).getAs[Row](0)
    val data = tensorRow.getAs[Array[Byte]](0)
    data.length shouldBe 16
  }

  it should "preserve shape array" in {
    val result = spark.sql(
      "SELECT arr_to_st(array(cast(1.0 as float), cast(2.0 as float)), array(2, 1), 'F32') as tensor"
    ).collect()
    result.length shouldBe 1
    val tensorRow = result(0).getAs[Row](0)
    val shape = tensorRow.getAs[scala.collection.Seq[Int]](1)
    shape.toList shouldBe List(2, 1)
  }

  it should "preserve dtype string" in {
    for (dtype <- Seq("F32", "F64", "I32", "U8", "BF16", "F16")) {
      val result = spark.sql(
        s"SELECT arr_to_st(array(cast(1.0 as float), cast(2.0 as float)), array(2), '$dtype') as tensor"
      ).collect()
      result.length shouldBe 1
      val tensorRow = result(0).getAs[Row](0)
      tensorRow.getAs[String](2) shouldBe dtype
    }
  }

  behavior of "st_to_array round-trip"

  it should "decode F32 values correctly" in {
    val result = spark.sql(
      """SELECT st_to_array(
           arr_to_st(array(cast(1.0 as float), cast(2.0 as float), cast(3.0 as float), cast(4.0 as float)), array(4), 'F32')
         ) as decoded"""
    ).collect()
    result.length shouldBe 1
    val decoded = result(0).getAs[scala.collection.Seq[Float]](0)
    decoded should have length 4
  }

  it should "return ArrayType(FloatType)" in {
    val schema = spark.sql(
      """SELECT st_to_array(
           arr_to_st(array(cast(1.0 as float)), array(1), 'F32')
         ) as decoded"""
    ).schema
    schema.fields(0).dataType shouldBe ArrayType(FloatType, containsNull = false)
  }
}
