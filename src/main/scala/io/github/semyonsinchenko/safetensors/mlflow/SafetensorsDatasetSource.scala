package io.github.semyonsinchenko.safetensors.mlflow

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession

/**
 * JVM-side utility for MLflow dataset lineage.
 *
 * Reads the dataset_manifest.json produced by a safetensors write operation
 * and serialises it as an MLflow-compatible dataset source JSON blob.
 *
 * This class is a JVM-side utility only. The Python log_dataset() function
 * reads dataset_manifest.json directly via standard Python file I/O and does
 * NOT call this Scala class.
 *
 * Usage (JVM / Scala):
 *   val source = SafetensorsDatasetSource.fromPath("/output/path")
 *   val json   = source.toJson()
 */
final class SafetensorsDatasetSource private (
  private val manifestMap: Map[String, Any],
  private val sourcePath:  String,
) {

  /**
   * Serialise the manifest as an MLflow dataset source JSON blob.
   * The blob contains the full manifest contents and the source path URI.
   */
  def toJson(): String = {
    val wrapper = Map(
      "source_type" -> "safetensors",
      "uri"         -> sourcePath,
      "manifest"    -> manifestMap,
    )
    SafetensorsDatasetSource.mapper.writeValueAsString(wrapper)
  }
}

object SafetensorsDatasetSource {

  private val mapper = new ObjectMapper().registerModule(DefaultScalaModule)

  /**
   * Read the dataset_manifest.json at the given output path and construct
   * a SafetensorsDatasetSource for JVM-side MLflow lineage tracking.
   *
   * @param outputPath  Root output directory containing dataset_manifest.json
   *                    (Hadoop-compatible URI).
   * @throws java.io.FileNotFoundException if dataset_manifest.json is absent.
   */
  def fromPath(outputPath: String): SafetensorsDatasetSource = {
    val spark = SparkSession.active
    val conf  = spark.sparkContext.hadoopConfiguration
    val manifestPath = new Path(outputPath, "dataset_manifest.json")
    val fs    = manifestPath.getFileSystem(conf)

    if (!fs.exists(manifestPath)) {
      throw new java.io.FileNotFoundException(
        s"dataset_manifest.json not found at: $manifestPath. " +
          "Run a safetensors write operation first.")
    }

    val in    = fs.open(manifestPath)
    val bytes = try { in.readAllBytes() } finally { in.close() }
    val manifestMap = mapper.readValue(new String(bytes, "UTF-8"), classOf[Map[String, Any]])

    new SafetensorsDatasetSource(manifestMap, outputPath)
  }
}
