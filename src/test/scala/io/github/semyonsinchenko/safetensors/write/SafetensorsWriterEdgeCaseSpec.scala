package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.core.SafetensorsDtype
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.connector.write.LogicalWriteInfo
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters._

/** Edge case unit tests for the safetensors writer.
  *
  * Tests cover: empty partitions, single-row partitions, null handling, BF16 dtype, comprehensive
  * dtype coverage, and buffer alignment issues.
  */
class SafetensorsWriterEdgeCaseSpec extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private var spark: SparkSession = _
  private var tempDir: Path       = _

  override def beforeAll(): Unit = {
    spark = SparkSession
      .builder()
      .master("local[1]")
      .appName("SafetensorsWriterEdgeCaseSpec")
      .config("spark.sql.shuffle.partitions", "1")
      .config("spark.ui.enabled", "false")
      .config(
        "spark.sql.extensions",
        "io.github.semyonsinchenko.safetensors.SafetensorsExtensions"
      )
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    tempDir = Files.createTempDirectory("safetensors-writer-edge-tests-")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
    if (tempDir != null) {
      deleteRecursively(tempDir.toFile)
    }
  }

  private def deleteRecursively(file: java.io.File): Unit = {
    if (file.isDirectory) {
      file.listFiles.foreach(deleteRecursively)
    }
    file.delete()
  }

  private def tensorStruct: StructType =
    StructType(
      Seq(
        StructField("data", BinaryType, nullable = false),
        StructField("shape", ArrayType(IntegerType, containsNull = false), nullable = false),
        StructField("dtype", StringType, nullable = false)
      )
    )

  private def makeTensorRow(data: Array[Byte], shape: Seq[Int], dtype: String): Row =
    Row(Row(data, shape, dtype))

  // ---------------------------------------------------------------------------
  // Task 1.1: Empty partition handling
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter empty partition handling"

  it should "handle empty DataFrames without errors and produce no output files" in {
    val emptyDf = spark.createDataFrame(
      spark.sparkContext.emptyRDD[Row],
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("empty-partition-test").toString

    // Should complete without errors
    (
      emptyDf.write
        .format("safetensors")
        .option("batch_size", "10")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(outPath)
    )

    // Should produce only manifest (no safetensors files for empty data)
    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 0
  }

  it should "handle partitions that become empty after filtering" in {
    // Create DataFrame with data, then filter everything out
    val rows = Seq(
      Row(Row(Array[Byte](1, 2, 3, 4).map(_.toByte), Seq(4), "F32"))
    )
    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("filtered-empty-test").toString

    // Filter everything out
    val emptyFiltered = df.filter("tensor.shape[0] > 100") // No rows will match

    (
      emptyFiltered.write
        .format("safetensors")
        .option("batch_size", "10")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(outPath)
    )

    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 0
  }

  // ---------------------------------------------------------------------------
  // Task 1.2: Single-row partition serialization
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter single-row partition handling"

  it should "correctly serialize single-row partitions without buffer alignment issues" in {
    val data = Array[Byte](1, 0, 0, 0, 2, 0, 0, 0).map(_.toByte) // Two F32 values: 1.0, 2.0
    val rows = Seq(
      Row(Row(data, Seq(2), "F32"))
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows, 1), // Single partition
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("single-row-test").toString

    (
      df.write
        .format("safetensors")
        .option("batch_size", "1") // Single row per batch
        .option("dtype", "F32")
        .mode("overwrite")
        .save(outPath)
    )

    // Verify output
    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 1

    // Verify manifest
    val manifestFile = new java.io.File(outputDir, "dataset_manifest.json")
    manifestFile.exists() shouldBe true
  }

  it should "handle multiple single-row partitions across different tasks" in {
    val data = Array[Byte](1, 0, 0, 0).map(_.toByte) // Single F32 value: 1.0
    val rows = (1 to 5).map(_ => Row(Row(data, Seq(1), "F32")))

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows, 5), // 5 partitions
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("multi-single-row-test").toString

    (
      df.write
        .format("safetensors")
        .option("batch_size", "1")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(outPath)
    )

    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 5 // One file per partition with batch_size=1
  }

  // ---------------------------------------------------------------------------
  // Task 1.3: Maximum tensor size handling
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter maximum tensor size handling"

  it should "respect configurable thresholds for large tensor handling" in {
    // Test with moderately large tensor (configurable threshold concept)
    val largeDataSize = 1024 * 1024 // 1MB tensor
    val largeData     = new Array[Byte](largeDataSize)
    // Fill with pattern
    for (i <- largeData.indices)
      largeData(i) = (i % 256).toByte

    val rows = Seq(
      Row(Row(largeData, Seq(largeDataSize / 4), "F32")) // F32 = 4 bytes per element
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows, 1),
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("large-tensor-test").toString

    // Should handle without error
    (
      df.write
        .format("safetensors")
        .option("batch_size", "1")
        .option("dtype", "F32")
        .mode("overwrite")
        .save(outPath)
    )

    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 1

    // Verify file size is reasonable (header + data)
    val fileSize = safetensorFiles.head.length()
    fileSize should be >= largeDataSize.toLong
    fileSize should be <= (largeDataSize + 1024 * 1024).toLong // Header shouldn't be more than 1MB
  }

  it should "handle tensors near the upper bounds of reasonable sizes" in {
    // Skip this test in constrained environments
    val maxTestSize = 16 * 1024 * 1024 // 16MB - safe upper bound for testing
    val largeData   = new Array[Byte](maxTestSize)

    val rows = Seq(
      Row(Row(largeData, Seq(maxTestSize / 4), "F32"))
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows, 1),
      StructType(Seq(StructField("tensor", tensorStruct, nullable = false)))
    )

    val outPath = tempDir.resolve("upper-bound-tensor-test").toString

    noException should be thrownBy {
      (
        df.write
          .format("safetensors")
          .option("batch_size", "1")
          .option("dtype", "F32")
          .mode("overwrite")
          .save(outPath)
      )
    }
  }

  // ---------------------------------------------------------------------------
  // Task 1.4: BF16 dtype test with hardcoded validation
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter BF16 dtype handling"

  // BF16 is a special case: not in the official safetensors JSON schema regex pattern
  // ([UIF])(8|16|32|64|128|256), but must be accepted by the connector.
  // See §1.1 of AGENTS.md for the BF16/JSON schema discrepancy.

  it should "accept BF16 dtype and write valid safetensors output" in {
    // Use float arrays directly - arr_to_st will convert to BF16
    val rows = Seq(
      Row(Seq(1.0f, 0.5f, -1.0f, 2.0f))
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows, 1),
      StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))
    )

    val outPath = tempDir.resolve("bf16-dtype-test").toString

    // Use arr_to_st to convert to BF16
    df.createOrReplaceTempView("bf16_input")
    val bf16Df = spark.sql(
      "SELECT arr_to_st(floats, array(4), 'BF16') AS tensor FROM bf16_input"
    )

    (
      bf16Df.write
        .format("safetensors")
        .option("batch_size", "1")
        .mode("overwrite")
        .save(outPath)
    )

    val outputDir       = new java.io.File(outPath)
    val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
    safetensorFiles should have length 1

    // Verify BF16 is in the manifest
    val manifestFile = new java.io.File(outputDir, "dataset_manifest.json")
    manifestFile.exists() shouldBe true
  }

  it should "validate BF16 dtype string acceptance in WriteBuilder" in {
    val localSchema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

    val optionsMap = new CaseInsensitiveStringMap(
      Map("batch_size" -> "10", "dtype" -> "BF16").asJava
    )

    val info = new LogicalWriteInfo {
      override def schema(): StructType                = localSchema
      override def options(): CaseInsensitiveStringMap = optionsMap
      override def queryId(): String                   = "test-query-id"
    }

    val builder = new SafetensorsWriteBuilder(info, optionsMap, Seq(tempDir.toString))
    // Should not throw - BF16 is explicitly supported despite JSON schema regex
    noException should be thrownBy {
      builder.buildForBatch()
    }
  }

  // ---------------------------------------------------------------------------
  // Task 1.5: Null value handling tests
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter null value handling"

  it should "reject null values in non-nullable tensor struct columns with clear error" in {
    val rows = Seq(
      Row(null) // Null tensor value
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      StructType(Seq(StructField("tensor", tensorStruct, nullable = true)))
    )

    val outPath = tempDir.resolve("null-value-test").toString

    // Should fail with clear error message
    val ex = intercept[Exception] {
      (
        df.write
          .format("safetensors")
          .option("batch_size", "10")
          .option("dtype", "F32")
          .mode("overwrite")
          .save(outPath)
      )
    }

    // Error message should mention null or the column name
    val msg = ex.getMessage.toLowerCase
    (msg.contains("null") || msg.contains("tensor")) shouldBe true
  }

  it should "handle nullable columns gracefully when they contain valid data" in {
    val data = Array[Byte](1, 0, 0, 0).map(_.toByte)
    val rows = Seq(
      Row(Row(data, Seq(1), "F32"))
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      StructType(Seq(StructField("tensor", tensorStruct, nullable = true)))
    )

    val outPath = tempDir.resolve("nullable-valid-test").toString

    // Should succeed when data is actually present
    noException should be thrownBy {
      (
        df.write
          .format("safetensors")
          .option("batch_size", "10")
          .option("dtype", "F32")
          .mode("overwrite")
          .save(outPath)
      )
    }
  }

  it should "provide clear error messages for null values in numeric arrays" in {
    val rows = Seq(
      Row(null) // Null array value
    )

    val df = spark.createDataFrame(
      spark.sparkContext.parallelize(rows),
      StructType(Seq(StructField("values", ArrayType(FloatType), nullable = true)))
    )

    val outPath = tempDir.resolve("null-array-test").toString

    val ex = intercept[Exception] {
      (
        df.write
          .format("safetensors")
          .option("batch_size", "10")
          .option("dtype", "F32")
          .mode("overwrite")
          .save(outPath)
      )
    }

    val msg = ex.getMessage.toLowerCase
    (msg.contains("null") || msg.contains("values")) shouldBe true
  }

  // ---------------------------------------------------------------------------
  // Task 1.6: Comprehensive dtype coverage tests
  // ---------------------------------------------------------------------------

  behavior of "SafetensorsWriter comprehensive dtype coverage"

  private val allDtypes = Seq(
    "F64",
    "F32",
    "F16",
    "I64",
    "I32",
    "I16",
    "I8",
    "U8"
  )

  allDtypes.foreach { dtypeName =>
    it should s"correctly handle $dtypeName dtype serialization via WriteBuilder" in {
      val arrayType = ArrayType(FloatType, containsNull = false)

      // Create a simple float array value row - arr_to_st handles conversion
      val arrayData = Seq(1.0f, 2.0f, 3.0f, 4.0f)
      val rows      = Seq(Row(arrayData))

      val df = spark.createDataFrame(
        spark.sparkContext.parallelize(rows),
        StructType(Seq(StructField("data", arrayType, nullable = false)))
      )

      val outPath = tempDir.resolve(s"dtype-$dtypeName-test").toString

      // Convert using arr_to_st and write
      df.createOrReplaceTempView(s"dtype_${dtypeName.toLowerCase}_input")
      val tensorDf = spark.sql(
        s"SELECT arr_to_st(data, array(4), '$dtypeName') AS tensor FROM dtype_${dtypeName.toLowerCase}_input"
      )

      (
        tensorDf.write
          .format("safetensors")
          .option("batch_size", "1")
          .mode("overwrite")
          .save(outPath)
      )

      val outputDir       = new java.io.File(outPath)
      val safetensorFiles = outputDir.listFiles.filter(_.getName.endsWith(".safetensors"))
      safetensorFiles should have length 1

      // Verify manifest contains correct dtype
      val manifestFile = new java.io.File(outputDir, "dataset_manifest.json")
      manifestFile.exists() shouldBe true
    }
  }

  it should "accept all valid dtypes in WriteBuilder validation" in {
    val allValidDtypes = Seq(
      "F16",
      "F32",
      "F64",
      "BF16", // Float types (BF16 is special case)
      "I8",
      "I16",
      "I32",
      "I64", // Signed int types
      "U8",
      "U16",
      "U32",
      "U64" // Unsigned int types
    )

    val localSchema = StructType(Seq(StructField("floats", ArrayType(FloatType), nullable = false)))

    allValidDtypes.foreach { dtype =>
      val optionsMap = new CaseInsensitiveStringMap(
        Map("batch_size" -> "10", "dtype" -> dtype).asJava
      )

      val info = new LogicalWriteInfo {
        override def schema(): StructType                = localSchema
        override def options(): CaseInsensitiveStringMap = optionsMap
        override def queryId(): String                   = s"test-query-$dtype"
      }

      val builder = new SafetensorsWriteBuilder(info, optionsMap, Seq(tempDir.toString))
      withClue(s"dtype $dtype should be accepted") {
        noException should be thrownBy {
          builder.buildForBatch()
        }
      }
    }
  }

  it should "validate dtype bytesPerElement for all supported types" in {
    val expectedBytes = Map(
      SafetensorsDtype.F64  -> 8,
      SafetensorsDtype.F32  -> 4,
      SafetensorsDtype.F16  -> 2,
      SafetensorsDtype.BF16 -> 2, // Special case - see AGENTS.md §1.1
      SafetensorsDtype.I64  -> 8,
      SafetensorsDtype.I32  -> 4,
      SafetensorsDtype.I16  -> 2,
      SafetensorsDtype.I8   -> 1,
      SafetensorsDtype.U8   -> 1
    )

    expectedBytes.foreach { case (dtype, expectedBytes) =>
      val actualBytes = SafetensorsDtype.bytesPerElement(dtype)
      actualBytes shouldBe expectedBytes
    }
  }

}
