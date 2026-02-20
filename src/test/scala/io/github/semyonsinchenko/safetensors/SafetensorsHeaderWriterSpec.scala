package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.{
  SafetensorsDtype,
  SafetensorsHeaderParser,
  SafetensorsHeaderWriter
}
import io.github.semyonsinchenko.safetensors.core.SafetensorsHeaderWriter.TensorDescriptor

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SafetensorsHeaderWriterSpec extends AnyFlatSpec with Matchers {

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private def roundTrip(tensors: Seq[TensorDescriptor]): core.SafetensorsHeader = {
    val buf = SafetensorsHeaderWriter.buildHeader(tensors)
    SafetensorsHeaderParser.parse(buf)
  }

  // ---------------------------------------------------------------------------
  // Basic round-trip tests
  // ---------------------------------------------------------------------------

  "SafetensorsHeaderWriter" should "produce a header that round-trips through the parser" in {
    val tensors = Seq(
      TensorDescriptor("weight", SafetensorsDtype.F32, Seq(3, 4), byteLength = 48L)
    )
    val header = roundTrip(tensors)

    header.tensors should have size 1
    val info = header.tensors("weight")
    info.dtype shouldBe SafetensorsDtype.F32
    info.shape shouldBe Seq(3, 4)
    info.dataOffsets.begin shouldBe 0L
    info.dataOffsets.end shouldBe 48L
  }

  it should "write multiple tensors with sequential data_offsets (no holes)" in {
    val tensors = Seq(
      TensorDescriptor("a", SafetensorsDtype.F32, Seq(2), byteLength = 8L),
      TensorDescriptor("b", SafetensorsDtype.I16, Seq(4), byteLength = 8L),
      TensorDescriptor("c", SafetensorsDtype.U8, Seq(16), byteLength = 16L)
    )
    val header = roundTrip(tensors)

    header.tensors should have size 3

    val a = header.tensors("a")
    val b = header.tensors("b")
    val c = header.tensors("c")

    a.dataOffsets.begin shouldBe 0L
    a.dataOffsets.end shouldBe 8L

    b.dataOffsets.begin shouldBe 8L
    b.dataOffsets.end shouldBe 16L

    c.dataOffsets.begin shouldBe 16L
    c.dataOffsets.end shouldBe 32L
  }

  it should "preserve tensor key insertion order in the round-tripped header" in {
    // Keys are deliberately out of alphabetical order to verify order preservation
    val tensors = Seq(
      TensorDescriptor("zebra", SafetensorsDtype.F32, Seq(1), byteLength = 4L),
      TensorDescriptor("apple", SafetensorsDtype.F32, Seq(1), byteLength = 4L),
      TensorDescriptor("mango", SafetensorsDtype.F32, Seq(1), byteLength = 4L)
    )
    val header = roundTrip(tensors)

    // tensorNames must preserve insertion order, not be sorted alphabetically
    header.tensorNames shouldBe Seq("zebra", "apple", "mango")
  }

  it should "handle BF16 dtype (special case outside JSON schema)" in {
    val tensors = Seq(
      // NOTE: BF16 is not in the JSON schema regex — see §1.1
      TensorDescriptor("emb", SafetensorsDtype.BF16, Seq(128), byteLength = 256L)
    )
    val header = roundTrip(tensors)
    header.tensors("emb").dtype shouldBe SafetensorsDtype.BF16
  }

  it should "handle a scalar (rank-0) tensor with empty shape" in {
    val tensors = Seq(
      TensorDescriptor("loss", SafetensorsDtype.F32, Seq.empty, byteLength = 4L)
    )
    val header = roundTrip(tensors)
    header.tensors("loss").shape shouldBe Seq.empty
  }

  it should "include __metadata__ when file metadata is supplied" in {
    val tensors = Seq(
      TensorDescriptor("w", SafetensorsDtype.F32, Seq(4), byteLength = 16L)
    )
    val meta   = Map("author" -> "test", "version" -> "1")
    val buf    = SafetensorsHeaderWriter.buildHeader(tensors, meta)
    val header = SafetensorsHeaderParser.parse(buf)

    header.metadata shouldBe meta
  }

  it should "write no __metadata__ key when fileMetadata is empty" in {
    val tensors = Seq(
      TensorDescriptor("w", SafetensorsDtype.F32, Seq(4), byteLength = 16L)
    )
    val header = roundTrip(tensors)
    header.metadata shouldBe Map.empty
  }

  it should "produce a binary header whose 8-byte prefix equals the JSON byte length" in {
    import java.nio.{ByteBuffer, ByteOrder}

    val tensors = Seq(
      TensorDescriptor("x", SafetensorsDtype.I32, Seq(2, 3), byteLength = 24L)
    )
    val buf = SafetensorsHeaderWriter.buildHeader(tensors)
    buf.order(ByteOrder.LITTLE_ENDIAN)

    val declaredSize = buf.getLong()
    val remaining    = buf.remaining()

    declaredSize shouldBe remaining.toLong
  }

  it should "reject duplicate tensor names" in {
    val tensors = Seq(
      TensorDescriptor("dup", SafetensorsDtype.F32, Seq(1), byteLength = 4L),
      TensorDescriptor("dup", SafetensorsDtype.F32, Seq(1), byteLength = 4L)
    )
    an[IllegalArgumentException] should be thrownBy
      SafetensorsHeaderWriter.buildHeader(tensors)
  }

}
