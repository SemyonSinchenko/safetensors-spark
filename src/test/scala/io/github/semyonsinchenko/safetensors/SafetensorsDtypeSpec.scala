package io.github.semyonsinchenko.safetensors

import io.github.semyonsinchenko.safetensors.core.SafetensorsDtype
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SafetensorsDtypeSpec extends AnyFlatSpec with Matchers {

  "SafetensorsDtype.fromString" should "parse all standard dtypes" in {
    val standard = Seq("F16", "F32", "F64", "U8", "I8", "U16", "I16", "U32", "I32", "U64", "I64")
    standard.foreach { s =>
      SafetensorsDtype.fromString(s) shouldBe a[Right[_, _]]
    }
  }

  // BF16 is a special case: not in the official JSON schema regex pattern
  // ([UIF])(8|16|32|64|128|256), but must be accepted by the connector.
  // See §1.1 of AGENTS.md for the BF16/JSON schema discrepancy.
  it should "accept BF16 as a special case outside the JSON schema regex" in {
    SafetensorsDtype.fromString("BF16") shouldBe Right(SafetensorsDtype.BF16)
  }

  it should "reject unknown dtype strings" in {
    SafetensorsDtype.fromString("F128") shouldBe a[Left[_, _]]
    SafetensorsDtype.fromString("INT32") shouldBe a[Left[_, _]]
    SafetensorsDtype.fromString("") shouldBe a[Left[_, _]]
  }

  "SafetensorsDtype.bytesPerElement" should "return correct byte widths" in {
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.U8) shouldBe 1
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.I8) shouldBe 1
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.F16) shouldBe 2
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.BF16) shouldBe 2
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.F32) shouldBe 4
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.F64) shouldBe 8
    SafetensorsDtype.bytesPerElement(SafetensorsDtype.I64) shouldBe 8
  }

}
