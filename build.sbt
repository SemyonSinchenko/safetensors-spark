// ---------------------------------------------------------------------------
// safetensors-spark — Apache Spark DataSource V2 for Hugging Face safetensors
// ---------------------------------------------------------------------------

// ---- Spark / Scala version matrix -----------------------------------------
// Override at build time with: sbt -DsparkVersion=4.0.1 compile
val sparkVersion = sys.props.getOrElse("sparkVersion", "4.1.0")

// Scala version is inferred from Spark major version to simplify
// future cross-version testing:
//   Spark 3.x -> Scala 2.12
//   Spark 4.x -> Scala 2.13
val scalaVersionValue =
  if (sparkVersion.startsWith("4.")) "2.13.14"
  else "2.12.18"

ThisBuild / scalaVersion := scalaVersionValue
ThisBuild / organization := "io.github.semyonsinchenko"
ThisBuild / version      := "0.1.0-SNAPSHOT"

// ---- Project definition ----------------------------------------------------
lazy val root = (project in file("."))
  .settings(
    name := "safetensors-spark",

    // Artifact coordinates: io.github.semyonsinchenko:safetensors-spark_2.13:<version>
    moduleName := "safetensors-spark",

    // Java 11+ compatibility
    javacOptions ++= Seq("-source", "11", "-target", "11"),
    scalacOptions ++= Seq(
      "-encoding", "utf8",
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Xfatal-warnings",
    ),

    // ---- Dependencies -------------------------------------------------------
    libraryDependencies ++= Seq(
      // Spark — provided at runtime by the cluster
      "org.apache.spark" %% "spark-core" % sparkVersion % Provided,
      "org.apache.spark" %% "spark-sql"  % sparkVersion % Provided,

      // JSON parsing for safetensors headers (included in Spark's classpath;
      // listed here to ensure IDE resolution)
      "com.fasterxml.jackson.core"   % "jackson-databind"          % "2.18.2" % Provided,
      "com.fasterxml.jackson.module" %% "jackson-module-scala"     % "2.18.2" % Provided,

      // ---- Test dependencies -----------------------------------------------
      "org.scalatest"  %% "scalatest"        % "3.2.19" % Test,
      "org.apache.spark" %% "spark-core"     % sparkVersion % Test,
      "org.apache.spark" %% "spark-sql"      % sparkVersion % Test,
    ),

    // ---- Test configuration -------------------------------------------------
    Test / fork := true,
    Test / javaOptions ++= Seq(
      "-Xmx2g",
      // Required for Spark on Java 17: open module access
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    ),

    // ---- Assembly / packaging -----------------------------------------------
    // Exclude Spark and Hadoop from the fat jar — they are provided by the cluster
    assembly / assemblyOption := (assembly / assemblyOption).value
      .withIncludeScala(false),
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "services", _*) => MergeStrategy.concat
      case PathList("META-INF", _*)              => MergeStrategy.discard
      case "reference.conf"                      => MergeStrategy.concat
      case _                                     => MergeStrategy.first
    },
  )
