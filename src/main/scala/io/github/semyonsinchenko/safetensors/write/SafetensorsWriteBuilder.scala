package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.core.TensorSchema
import io.github.semyonsinchenko.safetensors.util.Errors
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.write.{
  BatchWrite,
  LogicalWriteInfo,
  SupportsTruncate,
  WriteBuilder
}
import org.apache.spark.sql.types.{ArrayType, DataType, StructType}
import org.apache.spark.sql.util.CaseInsensitiveStringMap

import scala.util.control.NonFatal

/** WriteBuilder for the safetensors DataSource V2.
  *
  * All schema and type validation is performed eagerly here in buildForBatch() before any tasks are
  * launched, so that errors surface at plan time with clear messages (AnalysisException).
  *
  * Supports `mode("overwrite")` via the SupportsTruncate interface: when called, truncate() deletes
  * all existing .safetensors files from the output paths before returning a new builder instance
  * for buildForBatch().
  *
  * Accepted input column types (auto-detected per column):
  *   1. Tensor Struct (StructType with data/shape/dtype fields): raw bytes are written directly. 2.
  *      Numeric ArrayType (ArrayType with a numeric element type): the connector encodes array
  *      elements into raw bytes using the target dtype option. Non-numeric ArrayType (e.g.
  *      ArrayType(StringType)) is rejected.
  */
class SafetensorsWriteBuilder(
    private val info: LogicalWriteInfo,
    private val options: CaseInsensitiveStringMap,
    private val paths: Seq[String]
) extends WriteBuilder
    with SupportsTruncate {

  override def truncate(): WriteBuilder = {
    // Called by Spark when mode("overwrite") with a full-table truncate predicate.
    // Delete all existing .safetensors files from the output paths, then return a new builder
    // to proceed with the write (buildForBatch will be called next).
    deleteExistingSafetensorsFiles()
    this
  }

  override def buildForBatch(): BatchWrite = {
    // 1. Parse and validate write options (throws on invalid option combinations)
    val writeOptions = WriteOptions.parse(options)

    // 2. Validate the input DataFrame schema
    val inputSchema = info.schema()
    validateSchema(inputSchema, writeOptions)

    new SafetensorsBatchWrite(inputSchema, writeOptions, paths)
  }

  // ---------------------------------------------------------------------------
  // Schema validation
  // ---------------------------------------------------------------------------

  private def validateSchema(schema: StructType, opts: WriteOptions): Unit = {
    val columnsToWrite = opts.columns
      .map(cols => schema.fields.filter(f => cols.contains(f.name)))
      .getOrElse {
        // Exclude the name_col from tensor columns
        opts.namingStrategy match {
          case NameColStrategy(col) => schema.fields.filterNot(_.name == col)
          case _                    => schema.fields
        }
      }

    if (columnsToWrite.isEmpty) {
      throw Errors.analysisException(
        "No tensor columns found to write. " +
          "Check the 'columns' option and the DataFrame schema."
      )
    }

    columnsToWrite.foreach { field =>
      validateColumnType(field.name, field.dataType, opts)
    }

    // Validate name_col exists in schema if specified
    opts.namingStrategy match {
      case NameColStrategy(col) =>
        if (!schema.fieldNames.contains(col)) {
          throw Errors.analysisException(
            s"name_col '$col' does not exist in the DataFrame schema. " +
              s"Available columns: ${schema.fieldNames.mkString(", ")}"
          )
        }
      case _ =>
    }

    // dtype is required for numeric ArrayType input
    val hasArrayInput = columnsToWrite.exists(f => TensorSchema.isNumericArrayType(f.dataType))
    if (hasArrayInput && opts.dtype.isEmpty) {
      throw Errors.analysisException(
        "The 'dtype' option is required when writing numeric array columns. " +
          "Specify the target dtype (e.g. .option(\"dtype\", \"F32\"))."
      )
    }
  }

  private def validateColumnType(name: String, dt: DataType, opts: WriteOptions): Unit = dt match {
    case st: StructType if TensorSchema.isTensorStruct(st) =>
      // Valid: Tensor Struct input
      ()

    case at: ArrayType if TensorSchema.isNumericArrayType(at) =>
      // Valid: numeric array input — dtype option required (validated above)
      ()

    case at: ArrayType =>
      // Invalid: non-numeric array
      throw Errors.analysisException(
        s"Column '$name' has type ArrayType(${at.elementType.simpleString}) which is not a " +
          s"supported numeric type. Only ArrayType with numeric element types " +
          s"(${TensorSchema.numericElementTypes.map(_.simpleString).mkString(", ")}) " +
          s"are accepted as tensor input. " +
          s"Use arr_to_st() to convert, or pass a Tensor Struct column instead."
      )

    case other =>
      throw Errors.analysisException(
        s"Column '$name' has unsupported type '${other.simpleString}' for safetensors writing. " +
          s"Expected a Tensor Struct or a numeric ArrayType."
      )
  }

  // ---------------------------------------------------------------------------
  // Private helpers — truncate support (mode("overwrite"))
  // ---------------------------------------------------------------------------

  /** Delete all existing .safetensors files from the output paths.
    *
    * Called by truncate() when mode("overwrite") is used. This clears the directory before writing
    * new data, but preserves the manifest and index files (which will be overwritten by the write).
    */
  private def deleteExistingSafetensorsFiles(): Unit = {
    val hadoopConf = SparkSession.active.sparkContext.hadoopConfiguration
    paths.foreach { pathStr =>
      try {
        val path = new Path(pathStr)
        val fs   = FileSystem.get(path.toUri, hadoopConf)

        if (fs.exists(path)) {
          val fileStatus = fs.getFileStatus(path)
          if (fileStatus.isFile) {
            // Single file path: delete it if it's a .safetensors file
            if (path.getName.endsWith(".safetensors")) {
              fs.delete(path, false)
            }
          } else {
            // Directory: delete only .safetensors files (preserve manifest, index, etc.)
            val status = fs.listStatus(path)
            status.foreach { fileStatus =>
              if (fileStatus.getPath.getName.endsWith(".safetensors")) {
                fs.delete(fileStatus.getPath, false)
              }
            }
          }
        }
      } catch {
        case NonFatal(e) =>
          throw new RuntimeException(
            s"Failed to delete existing .safetensors files from path $pathStr: ${e.getMessage}",
            e
          )
      }
    }
  }

}
