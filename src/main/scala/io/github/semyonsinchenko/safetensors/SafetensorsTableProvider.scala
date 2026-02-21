package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.{SafetensorsHeaderParser, TensorSchema}
import io.github.semyonsinchenko.safetensors.util.Errors
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.catalog.{Table, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import java.util
import scala.jdk.CollectionConverters._
import scala.util.control.NonFatal

/** Entry point for the "safetensors" DataSource V2 format.
  *
  * Registered via META-INF/services/org.apache.spark.sql.sources.DataSourceRegister (short name:
  * "safetensors").
  *
  * Schema resolution order:
  *   1. User-provided schema via .schema(...) on the DataFrameReader. 2. Inference when
  *      .option("inferSchema", "true") is set — reads the header of the first file only; all files
  *      are assumed to share the same schema. 3. Otherwise, throws AnalysisException with a clear
  *      message.
  */
class SafetensorsTableProvider extends TableProvider with DataSourceRegister {

  override def shortName(): String = "safetensors"

  override def supportsExternalMetadata(): Boolean = true

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    val inferSchema = options.getBoolean("inferSchema", false)
    if (!inferSchema) {
      throw Errors.analysisException(
        "No schema provided for the safetensors format. " +
          "Either call .schema(...) on the DataFrameReader to supply an explicit schema, " +
          "or set .option(\"inferSchema\", \"true\") to infer it from the first file."
      )
    }
    inferSchemaFromFiles(options)
  }

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String]
  ): Table = {
    val opts  = new CaseInsensitiveStringMap(properties)
    val paths = resolvePaths(opts)
    new SafetensorsTable(schema, opts, paths)
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  private def resolvePaths(options: CaseInsensitiveStringMap): Seq[String] =
    Option(options.get("path"))
      .orElse(Option(options.get("paths")))
      .map(_.split(",").map(_.trim).toSeq)
      .getOrElse(Seq.empty)

  /** Infer schema by reading _tensor_index.parquet if available (§3.2), otherwise from the header
    * of the first .safetensors file. Each tensor key becomes one Spark column of type Tensor Struct
    * (see TensorSchema).
    */
  private def inferSchemaFromFiles(options: CaseInsensitiveStringMap): StructType = {
    val spark = SparkSession.active
    val paths = resolvePaths(options)
    require(paths.nonEmpty, "No path specified for safetensors data source")

    val rootPath = paths.head

    // Check for _tensor_index.parquet first (§3.2)
    val indexPath  = new Path(rootPath, "_tensor_index.parquet")
    val hadoopConf = spark.sparkContext.hadoopConfiguration
    val fs = rootPath.split(":") match {
      case Array(scheme, rest) if scheme.matches("[a-zA-Z]+") =>
        // URI with scheme (hdfs://, s3a://, etc.)
        new Path(rootPath).getFileSystem(hadoopConf)
      case _ =>
        // Local path
        new Path(rootPath).getFileSystem(hadoopConf)
    }

    if (fs.exists(indexPath)) {
      // Read the index to extract distinct tensor keys
      try {
        val indexDf = spark.read.parquet(indexPath.toString)
        val tensorKeys = indexDf
          .select("tensor_key")
          .distinct()
          .collect()
          .map(_.getString(0))
          .sorted
          .toSeq

        if (tensorKeys.nonEmpty) {
          val fields: Seq[StructField] = tensorKeys.map { name =>
            StructField(name, TensorSchema.schema, nullable = false)
          }
          return StructType(fields)
        }
      } catch {
        case NonFatal(_) =>
        // Fall through to file-based inference if index read fails
      }
    }

    // Fall back to reading the first .safetensors file header
    val firstFile = findFirstSafetensorsFile(spark, rootPath)
      .getOrElse(
        throw Errors.analysisException(s"No .safetensors files found under: $rootPath")
      )

    // Open the file and parse its header
    val header = {
      val in     = fs.open(new Path(firstFile))
      val length = fs.getFileStatus(new Path(firstFile)).getLen
      val bytes =
        new Array[Byte](math.min(length, 256 * 1024).toInt) // read up to 256 KB for header
      val nRead = in.read(bytes)
      in.close()
      val buf = java.nio.ByteBuffer.wrap(bytes, 0, nRead)
      SafetensorsHeaderParser.parse(buf)
    }

    // Build a wide/columnar schema: one Tensor Struct column per tensor key
    val fields: Seq[StructField] = header.tensors.keys.toSeq.sorted.map { name =>
      StructField(name, TensorSchema.schema, nullable = false)
    }
    StructType(fields)
  }

  private def findFirstSafetensorsFile(spark: SparkSession, path: String): Option[String] = {
    val hadoopPath = new Path(path)
    val fs         = hadoopPath.getFileSystem(spark.sparkContext.hadoopConfiguration)

    if (!fs.exists(hadoopPath)) return None

    val status = fs.getFileStatus(hadoopPath)
    if (!status.isDirectory) {
      // Direct file path
      if (path.endsWith(".safetensors")) Some(path) else None
    } else {
      // Directory: find first .safetensors file (non-recursive, sorted)
      fs.listStatus(hadoopPath)
        .filter(s => !s.isDirectory && s.getPath.getName.endsWith(".safetensors"))
        .sortBy(_.getPath.getName)
        .headOption
        .map(_.getPath.toString)
    }
  }

}
