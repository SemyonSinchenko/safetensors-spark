package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.{SafetensorsDtype, SafetensorsHeaderParser}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets

class SafetensorsHeaderParserSpec extends AnyFlatSpec with Matchers {

  /** Build a minimal valid safetensors header buffer for testing. */
  private def buildTestHeader(json: String): ByteBuffer = {
    val jsonBytes   = json.getBytes(StandardCharsets.UTF_8)
    val headerSize  = jsonBytes.length.toLong
    val buf         = ByteBuffer.allocate(8 + jsonBytes.length)
    buf.order(ByteOrder.LITTLE_ENDIAN)
    buf.putLong(headerSize)
    buf.put(jsonBytes)
    buf.flip()
    buf
  }

  "SafetensorsHeaderParser" should "parse a simple single-tensor header" in {
    val json =
      """{"weight": {"dtype": "F32", "shape": [3, 4], "data_offsets": [0, 48]}}"""
    val header = SafetensorsHeaderParser.parse(buildTestHeader(json))

    header.tensors should have size 1
    val info = header.tensors("weight")
    info.dtype            shouldBe SafetensorsDtype.F32
    info.shape            shouldBe Seq(3, 4)
    info.dataOffsets.begin shouldBe 0L
    info.dataOffsets.end   shouldBe 48L
    info.dataOffsets.byteSize shouldBe 48L
  }

  it should "parse a header with BF16 dtype (special case outside JSON schema)" in {
    val json =
      """{"emb": {"dtype": "BF16", "shape": [128], "data_offsets": [0, 256]}}"""
    val header = SafetensorsHeaderParser.parse(buildTestHeader(json))
    header.tensors("emb").dtype shouldBe SafetensorsDtype.BF16
  }

  it should "parse the __metadata__ key" in {
    val json =
      """{"__metadata__": {"author": "test"}, "x": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]}}"""
    val header = SafetensorsHeaderParser.parse(buildTestHeader(json))
    header.metadata              shouldBe Map("author" -> "test")
    header.tensors("x").dtype    shouldBe SafetensorsDtype.I32
  }

  it should "expose the correct byteBufferOffset" in {
    val json    = """{"a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}"""
    val jsonLen = json.getBytes(StandardCharsets.UTF_8).length
    val header  = SafetensorsHeaderParser.parse(buildTestHeader(json))
    header.headerSize       shouldBe jsonLen.toLong
    header.byteBufferOffset shouldBe 8L + jsonLen
  }

  it should "reject a header that does not start with '{'" in {
    val bad = buildTestHeader("not-json")
    an[IllegalArgumentException] should be thrownBy SafetensorsHeaderParser.parse(bad)
  }

  it should "reject a buffer with fewer than 8 bytes" in {
    val tiny = ByteBuffer.allocate(4)
    an[Exception] should be thrownBy SafetensorsHeaderParser.parse(tiny)
  }

  it should "parse a scalar (0-rank) tensor with empty shape" in {
    val json = """{"scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}"""
    val header = SafetensorsHeaderParser.parse(buildTestHeader(json))
    header.tensors("scalar").shape shouldBe Seq.empty
  }
}
