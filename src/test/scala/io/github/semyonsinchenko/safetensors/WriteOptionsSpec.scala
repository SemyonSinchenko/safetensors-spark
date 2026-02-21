package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.SafetensorsDtype
import io.github.semyonsinchenko.safetensors.write._

import org.apache.spark.sql.util.CaseInsensitiveStringMap

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.util.{HashMap => JHashMap}

class WriteOptionsSpec extends AnyFlatSpec with Matchers {

  // ---------------------------------------------------------------------------
  // Helper
  // ---------------------------------------------------------------------------

  private def opts(pairs: (String, String)*): CaseInsensitiveStringMap = {
    val m = new JHashMap[String, String]()
    pairs.foreach { case (k, v) => m.put(k, v) }
    new CaseInsensitiveStringMap(m)
  }

  // ---------------------------------------------------------------------------
  // Naming strategy
  // ---------------------------------------------------------------------------

  "WriteOptions.parse" should "parse batch_size strategy" in {
    val o = WriteOptions.parse(opts("batch_size" -> "32"))
    o.namingStrategy shouldBe BatchSizeStrategy(32)
  }

  it should "parse name_col strategy" in {
    val o = WriteOptions.parse(opts("name_col" -> "key"))
    o.namingStrategy shouldBe NameColStrategy("key")
  }

  it should "reject both batch_size and name_col specified together" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "8", "name_col" -> "key"))
  }

  it should "reject neither batch_size nor name_col" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("dtype" -> "F32"))
  }

  it should "reject batch_size of zero" in {
    an[Exception] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "0"))
  }

  it should "reject batch_size of negative value" in {
    an[Exception] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "-1"))
  }

  // ---------------------------------------------------------------------------
  // dtype
  // ---------------------------------------------------------------------------

  it should "parse a valid dtype" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "dtype" -> "F32"))
    o.dtype shouldBe Some(SafetensorsDtype.F32)
  }

  it should "parse BF16 dtype (special case)" in {
    // NOTE: BF16 is not in the JSON schema regex — see §1.1
    val o = WriteOptions.parse(opts("batch_size" -> "1", "dtype" -> "BF16"))
    o.dtype shouldBe Some(SafetensorsDtype.BF16)
  }

  it should "reject an unknown dtype" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "1", "dtype" -> "FLOAT32"))
  }

  it should "leave dtype as None when not specified" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.dtype shouldBe None
  }

  // ---------------------------------------------------------------------------
  // tail_strategy (batch mode)
  // ---------------------------------------------------------------------------

  it should "default tail_strategy to DropTail" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.tailStrategy shouldBe DropTail
  }

  it should "parse tail_strategy=drop" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "tail_strategy" -> "drop"))
    o.tailStrategy shouldBe DropTail
  }

  it should "parse tail_strategy=pad" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "tail_strategy" -> "pad"))
    o.tailStrategy shouldBe PadWithZeros
  }

  it should "parse tail_strategy=write" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "tail_strategy" -> "write"))
    o.tailStrategy shouldBe WriteAsIs
  }

  it should "reject an unknown tail_strategy" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "1", "tail_strategy" -> "keep"))
  }

  // ---------------------------------------------------------------------------
  // target_shard_size_mb (KV mode)
  // ---------------------------------------------------------------------------

  it should "use the default shard size when not specified" in {
    val o = WriteOptions.parse(opts("name_col" -> "key"))
    o.targetShardSizeMb shouldBe WriteOptions.DEFAULT_TARGET_SHARD_SIZE_MB
  }

  it should "accept a shard size within the valid range in KV mode" in {
    val o = WriteOptions.parse(opts("name_col" -> "key", "target_shard_size_mb" -> "100"))
    o.targetShardSizeMb shouldBe 100
  }

  it should "reject a shard size below the minimum" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("name_col" -> "key", "target_shard_size_mb" -> "49"))
  }

  it should "reject a shard size above the maximum" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("name_col" -> "key", "target_shard_size_mb" -> "1001"))
  }

  it should "accept the minimum shard size exactly" in {
    val o = WriteOptions.parse(opts("name_col" -> "key", "target_shard_size_mb" -> "50"))
    o.targetShardSizeMb shouldBe 50
  }

  it should "accept the maximum shard size exactly" in {
    val o = WriteOptions.parse(opts("name_col" -> "key", "target_shard_size_mb" -> "1000"))
    o.targetShardSizeMb shouldBe 1000
  }

  // ---------------------------------------------------------------------------
  // duplicatesStrategy
  // ---------------------------------------------------------------------------

  it should "default to FailOnDuplicate" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.duplicatesStrategy shouldBe FailOnDuplicate
  }

  it should "parse 'lastWin' strategy (case-insensitive)" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "duplicatesStrategy" -> "lastWin"))
    o.duplicatesStrategy shouldBe LastWinOnDuplicate
  }

  it should "parse 'fail' strategy" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "duplicatesStrategy" -> "fail"))
    o.duplicatesStrategy shouldBe FailOnDuplicate
  }

  it should "reject an unknown duplicatesStrategy" in {
    an[IllegalArgumentException] should be thrownBy
      WriteOptions.parse(opts("batch_size" -> "1", "duplicatesStrategy" -> "IGNORE"))
  }

  // ---------------------------------------------------------------------------
  // generate_index
  // ---------------------------------------------------------------------------

  it should "default generate_index to false" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.generateIndex shouldBe false
  }

  it should "parse generate_index=true" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "generate_index" -> "true"))
    o.generateIndex shouldBe true
  }

  // ---------------------------------------------------------------------------
  // columns option
  // ---------------------------------------------------------------------------

  it should "leave columns as None when not specified" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.columns shouldBe None
  }

  it should "parse a comma-separated list of columns" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "columns" -> "image, label, weight"))
    o.columns shouldBe Some(Seq("image", "label", "weight"))
  }

  // ---------------------------------------------------------------------------
  // shapes option
  // ---------------------------------------------------------------------------

  it should "leave shapes as empty map when not specified" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.shapes shouldBe Map.empty
  }

  it should "parse a JSON shapes option" in {
    val shapesJson = """{"image": [3, 224, 224], "label": [1]}"""
    val o          = WriteOptions.parse(opts("batch_size" -> "1", "shapes" -> shapesJson))
    o.shapes shouldBe Map("image" -> Seq(3, 224, 224), "label" -> Seq(1))
  }

  // ---------------------------------------------------------------------------
  // kv_separator option
  // ---------------------------------------------------------------------------

  it should "default kv_separator to '__'" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1"))
    o.kvSeparator shouldBe "__"
  }

  it should "parse a custom kv_separator" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "kv_separator" -> "/"))
    o.kvSeparator shouldBe "/"
  }

  it should "accept an empty kv_separator" in {
    val o = WriteOptions.parse(opts("batch_size" -> "1", "kv_separator" -> ""))
    o.kvSeparator shouldBe ""
  }

}
