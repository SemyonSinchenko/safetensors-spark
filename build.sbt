// ---------------------------------------------------------------------------
// safetensors-spark — Apache Spark DataSource V2 for Hugging Face safetensors
// ---------------------------------------------------------------------------

// ---- Spark / Scala version matrix -----------------------------------------
// Override at build time with: sbt -DsparkVersion=4.0.1 compile
val sparkVersion = sys.props.getOrElse("sparkVersion", "4.1.0")

// Scala version is inferred from Spark major version to simplify
// future cross-version testing:
//   Spark 4.x -> Scala 2.13
val scalaVersionValue =
  if (sparkVersion.startsWith("4.")) "2.13.18"
  else "2.13.18"

// Spark minor version shim selector: "spark-4.0", "spark-4.1", etc.
// Extracts the first two version components, e.g. "4.0.1" -> "spark-4.0"
val sparkShimDir = "spark-" + sparkVersion.split("\\.").take(2).mkString(".")

// Artifact name suffix: "4.0", "4.1", etc.
val sparkMinorVersion = sparkVersion.split("\\.").take(2).mkString(".")

ThisBuild / scalaVersion := scalaVersionValue
ThisBuild / organization := "io.github.semyonsinchenko"
ThisBuild / version      := "0.1.0-SNAPSHOT"

// ---- POM / Maven Central metadata (required by sbt-ci-release) ------------
ThisBuild / homepage := Some(url("https://github.com/SemyonSinchenko/safetensors-spark"))
ThisBuild / licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / versionScheme := Some("semver-spec")
ThisBuild / scmInfo := Some(
  ScmInfo(
    browseUrl  = url("https://github.com/SemyonSinchenko/safetensors-spark"),
    connection = "scm:git:git@github.com:SemyonSinchenko/safetensors-spark.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id    = "SemyonSinchenko",
    name  = "Sem",
    email = "ssinchenko@apache.org",
    url   = url("https://github.com/SemyonSinchenko")
  )
)

// ---- Project definition ----------------------------------------------------
lazy val root = (project in file("."))
  .settings(
    name := s"safetensors-spark-$sparkMinorVersion",

    // Artifact coordinates: io.github.semyonsinchenko:safetensors-spark-4.1_2.13:<version>
    moduleName := s"safetensors-spark-$sparkMinorVersion",

    // Java 11+ compatibility
    javacOptions ++= Seq("-source", "11", "-target", "11"),
    scalacOptions ++= Seq(
      "-encoding", "utf8",
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Xfatal-warnings",
    ),

    // ---- Shim source directory ----------------------------------------------
    // Add the Spark-version-specific shim directory so that the correct
    // implementation of Errors (and any future shims) is picked up at compile
    // time, while the shared sources in src/main/scala remain unchanged.
    Compile / unmanagedSourceDirectories +=
      baseDirectory.value / "src" / "main" / s"scala-$sparkShimDir",

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
      "org.scalatest"    %% "scalatest"  % "3.2.19"     % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion % Test,
      "org.apache.spark" %% "spark-sql"  % sparkVersion % Test,
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
