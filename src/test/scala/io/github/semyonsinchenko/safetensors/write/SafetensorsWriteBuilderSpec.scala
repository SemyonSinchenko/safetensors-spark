package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.util.Errors
import org.apache.spark.sql.connector.write.LogicalWriteInfo
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.jdk.CollectionConverters._

/** Unit tests for SafetensorsWriteBuilder schema validation.
  *
  * Verifies that all schema validation happens eagerly at buildForBatch() time,
  * throwing AnalysisException with clear messages for invalid inputs.
  *
  * Tests validate:
  *   - Non-numeric ArrayType rejection
  *   - Missing dtype for numeric array input
  *   - name_col column existence validation
  *   - Mutual exclusion of batch_size and name_col options
  *   - Empty column list rejection
  */
class SafetensorsWriteBuilderSpec extends AnyFlatSpec with Matchers {

  private def tensorStruct: StructType =
    StructType(
      Seq(
        StructField("data", BinaryType, nullable = false),
        StructField("shape", ArrayType(IntegerType, containsNull = false), nullable = false),
        StructField("dtype", StringType, nullable = false)
      )
    )

  private def createWriteBuilder(
      inputSchema: StructType,
      options: Map[String, String]
  ): SafetensorsWriteBuilder = {
    val optionsMap = new CaseInsensitiveStringMap(options.asJava)
    val info = new LogicalWriteInfo {
      override def schema(): StructType = inputSchema
      override def options(): CaseInsensitiveStringMap = optionsMap
      override def queryId(): String = "test-query-id"
    }
    new SafetensorsWriteBuilder(info, optionsMap, Seq("/tmp/out"))
  }

  behavior of "SafetensorsWriteBuilder schema validation"

  it should "accept Tensor Struct input columns" in {
    val schema = StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10"))
    // Should not throw
    builder.buildForBatch()
  }

  it should "accept numeric array input with dtype option" in {
    val schema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "dtype" -> "F32"))
    // Should not throw
    builder.buildForBatch()
  }

  it should "accept all numeric element types" in {
    for (elementType <- Seq(FloatType, DoubleType, IntegerType, LongType, ShortType, ByteType)) {
      val schema = StructType(Seq(StructField("array", ArrayType(elementType), nullable = false)))

      val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "dtype" -> "F32"))
      // Should not throw
      builder.buildForBatch()
    }
  }

  it should "reject non-numeric ArrayType with clear message" in {
    val schema = StructType(Seq(StructField("strings", ArrayType(StringType), nullable = false)))

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "dtype" -> "F32"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("numeric")
    ex.getMessage should include("strings")
  }

  it should "reject numeric array without dtype option" in {
    val schema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("dtype")
    ex.getMessage.toLowerCase should include("required")
  }

  it should "reject unsupported column types" in {
    val schema = StructType(
      Seq(
        StructField("mapdata", MapType(StringType, IntegerType), nullable = false)
      )
    )

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("unsupported")
    ex.getMessage should include("mapdata")
  }

  behavior of "Write option validation"

  it should "reject batch_size and name_col together" in {
    val schema = StructType(
      Seq(
        StructField("tensor", tensorStruct, nullable = false),
        StructField("key", StringType, nullable = false)
      )
    )

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "name_col" -> "key"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    // Should fail at option parsing time, not here
    // (this test verifies the option validation logic works upstream)
  }

  it should "reject missing batch_size and name_col together" in {
    val schema = StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))

    val builder = createWriteBuilder(schema, Map.empty)

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    // Should fail at option parsing time
  }

  behavior of "name_col validation"

  it should "validate that name_col column exists in schema" in {
    val schema = StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))

    val builder = createWriteBuilder(
      schema,
      Map("name_col" -> "nonexistent_column", "dtype" -> "F32")
    )

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("name_col")
    ex.getMessage should include("does not exist")
    ex.getMessage should include("nonexistent_column")
  }

  it should "accept valid name_col column" in {
    val schema = StructType(
      Seq(
        StructField("tensor", tensorStruct, nullable = false),
        StructField("key", StringType, nullable = false)
      )
    )

    val builder = createWriteBuilder(schema, Map("name_col" -> "key", "dtype" -> "F32"))
    // Should not throw
    builder.buildForBatch()
  }

  behavior of "Column filtering"

  it should "respect columns option" in {
    val schema = StructType(
      Seq(
        StructField("tensor1", tensorStruct, nullable = false),
        StructField("tensor2", tensorStruct, nullable = false),
        StructField("other", StringType, nullable = false)
      )
    )

    val builder =
      createWriteBuilder(schema, Map("batch_size" -> "10", "columns" -> "tensor1"))
    // Should not throw - only tensor1 will be written
    builder.buildForBatch()
  }

  it should "reject when columns option results in no tensors" in {
    val schema = StructType(
      Seq(
        StructField("tensor1", tensorStruct, nullable = false),
        StructField("other", StringType, nullable = false)
      )
    )

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "columns" -> "other"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("unsupported")
  }

  behavior of "Dtype validation"

  it should "accept all valid dtypes" in {
    for (dtype <- Seq("F16", "F32", "F64", "BF16", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64")) {
      val schema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

      val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "dtype" -> dtype))
      // Should not throw
      builder.buildForBatch()
    }
  }

  it should "reject invalid dtype" in {
    val schema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

    val builder = createWriteBuilder(schema, Map("batch_size" -> "10", "dtype" -> "INVALID"))

    val ex = intercept[Exception] {
      builder.buildForBatch()
    }

    ex.getMessage should include("dtype")
  }
}
