package io.github.semyonsinchenko.safetensors.write

import io.github.semyonsinchenko.safetensors.core.SafetensorsDtype

import org.apache.spark.sql.util.CaseInsensitiveStringMap

/** Parsed and validated write options for the safetensors DataSource V2 writer.
  *
  * All validation is performed at construction time so that errors surface eagerly in
  * WriteBuilder.buildForBatch() before any tasks are launched.
  */
final case class WriteOptions(
    /** Columns to serialize (None = all columns). */
    columns: Option[Seq[String]],

    /** Per-sample shape overrides keyed by column name. */
    shapes: Map[String, Seq[Int]],

    /** Target output dtype (applied to all tensors). */
    dtype: Option[SafetensorsDtype],

    /** Naming strategy — exactly one of batchSize or nameCol must be set. */
    namingStrategy: NamingStrategy,

    /** Whether to write _tensor_index.parquet at the output root. */
    generateIndex: Boolean,

    /** Target shard file size in megabytes (50–1000). Used in KV mode only. */
    targetShardSizeMb: Int,

    /** How to handle duplicate tensor keys in name_col mode. */
    duplicatesStrategy: DuplicatesStrategy,

    /** Separator between name_col value and column name in multi-column KV mode. */
    kvSeparator: String,

    /** How to handle the last incomplete batch in batch_size mode.
      *
      * When the number of rows in a partition is not a multiple of batch_size, the final batch will
      * have fewer rows. This option controls what to do with that tail:
      *   - DropTail: discard the incomplete batch.
      *   - PadWithZeros: zero-pad the incomplete batch to reach exactly batch_size rows.
      *   - WriteAsIs: write the incomplete batch as-is (smaller leading dimension, default).
      */
    tailStrategy: TailStrategy
)

sealed trait NamingStrategy
case class BatchSizeStrategy(batchSize: Int) extends NamingStrategy
case class NameColStrategy(nameCol: String)  extends NamingStrategy

sealed trait DuplicatesStrategy
case object FailOnDuplicate    extends DuplicatesStrategy
case object LastWinOnDuplicate extends DuplicatesStrategy

/** Controls how the incomplete tail batch is handled in batch_size mode. */
sealed trait TailStrategy
case object DropTail     extends TailStrategy
case object PadWithZeros extends TailStrategy
case object WriteAsIs    extends TailStrategy

object WriteOptions {

  val DEFAULT_TARGET_SHARD_SIZE_MB = 300
  val MIN_SHARD_SIZE_MB            = 50
  val MAX_SHARD_SIZE_MB            = 1000

  /** Parse and validate write options from a CaseInsensitiveStringMap. Throws
    * IllegalArgumentException for invalid combinations.
    */
  def parse(options: CaseInsensitiveStringMap): WriteOptions = {
    // columns
    val columns = Option(options.get("columns"))
      .map(_.split(",").map(_.trim).filter(_.nonEmpty).toSeq)

    // shapes (JSON object: {"col": [dim0, dim1, ...]})
    val shapes: Map[String, Seq[Int]] = Option(options.get("shapes")) match {
      case None => Map.empty
      case Some(json) =>
        import com.fasterxml.jackson.databind.ObjectMapper
        import com.fasterxml.jackson.module.scala.DefaultScalaModule
        val mapper = new ObjectMapper().registerModule(DefaultScalaModule)
        mapper
          .readValue(json, classOf[Map[String, Any]])
          .map { case (k, v) =>
            k -> v.asInstanceOf[Seq[Any]].map {
              case n: Int    => n
              case n: Long   => n.toInt
              case n: Number => n.intValue()
            }
          }
    }

    // dtype
    val dtype: Option[SafetensorsDtype] = Option(options.get("dtype")).map { s =>
      SafetensorsDtype.fromString(s).fold(err => throw new IllegalArgumentException(err), identity)
    }

    // naming strategy (mutually exclusive: batch_size vs name_col)
    val batchSizeOpt = Option(options.get("batch_size")).map { s =>
      val n = s.toInt
      require(n > 0, s"batch_size must be positive, got: $n")
      n
    }
    val nameColOpt = Option(options.get("name_col")).filter(_.nonEmpty)

    val namingStrategy: NamingStrategy = (batchSizeOpt, nameColOpt) match {
      case (Some(bs), None)  => BatchSizeStrategy(bs)
      case (None, Some(col)) => NameColStrategy(col)
      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          "Options 'batch_size' and 'name_col' are mutually exclusive — specify at most one."
        )
      case (None, None) =>
        throw new IllegalArgumentException(
          "One of 'batch_size' or 'name_col' must be specified for the safetensors writer."
        )
    }

    // duplicatesStrategy
    val duplicatesStrategy: DuplicatesStrategy =
      Option(options.get("duplicatesStrategy")).getOrElse("fail").toLowerCase match {
        case "fail"    => FailOnDuplicate
        case "lastwin" => LastWinOnDuplicate
        case other =>
          throw new IllegalArgumentException(
            s"Unknown duplicatesStrategy '$other'. Valid values: fail, lastWin"
          )
      }

    // target shard size (used in KV mode only)
    val targetShardSizeMb: Int =
      Option(options.get("target_shard_size_mb"))
        .map(_.toInt)
        .getOrElse(DEFAULT_TARGET_SHARD_SIZE_MB)
    require(
      targetShardSizeMb >= MIN_SHARD_SIZE_MB && targetShardSizeMb <= MAX_SHARD_SIZE_MB,
      s"target_shard_size_mb must be between $MIN_SHARD_SIZE_MB and $MAX_SHARD_SIZE_MB, " +
        s"got: $targetShardSizeMb"
    )

    // generate_index
    val generateIndex = options.getBoolean("generate_index", false)

    // kv_separator (default: "__", any string accepted including empty)
    val kvSeparator = Option(options.get("kv_separator")).getOrElse("__")

    // tail_strategy (batch_size mode only; ignored in KV mode)
    val tailStrategy: TailStrategy =
      Option(options.get("tail_strategy")).getOrElse("write").toLowerCase match {
        case "drop"  => DropTail
        case "pad"   => PadWithZeros
        case "write" => WriteAsIs
        case other =>
          throw new IllegalArgumentException(
            s"Unknown tail_strategy '$other'. Valid values: drop, pad, write"
          )
      }

    WriteOptions(
      columns = columns,
      shapes = shapes,
      dtype = dtype,
      namingStrategy = namingStrategy,
      generateIndex = generateIndex,
      targetShardSizeMb = targetShardSizeMb,
      duplicatesStrategy = duplicatesStrategy,
      kvSeparator = kvSeparator,
      tailStrategy = tailStrategy
    )
  }

}
