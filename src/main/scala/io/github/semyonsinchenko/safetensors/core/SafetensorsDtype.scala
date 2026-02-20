package io.github.semyonsinchenko.safetensors.core

/** Enumeration of all valid safetensors dtype strings.
  *
  * NOTE on BF16: The official format/safetensors.schema.json dtype pattern is
  * `([UIF])(8|16|32|64|128|256)`, which does NOT match "BF16". BF16 is a known special case handled
  * outside the JSON schema by the safetensors library. This connector hardcodes BF16 as a valid
  * dtype, bypassing the schema regex. See §1.1 of the project specification (AGENTS.md) for
  * details.
  */
sealed abstract class SafetensorsDtype(val name: String) {
  override def toString: String = name
}

object SafetensorsDtype {
  case object F16 extends SafetensorsDtype("F16")
  case object F32 extends SafetensorsDtype("F32")
  case object F64 extends SafetensorsDtype("F64")
  // BF16 is a special case — not present in the official JSON schema pattern
  // ([UIF])(8|16|32|64|128|256), but is a valid dtype in the safetensors library.
  case object BF16 extends SafetensorsDtype("BF16")
  case object U8   extends SafetensorsDtype("U8")
  case object I8   extends SafetensorsDtype("I8")
  case object U16  extends SafetensorsDtype("U16")
  case object I16  extends SafetensorsDtype("I16")
  case object U32  extends SafetensorsDtype("U32")
  case object I32  extends SafetensorsDtype("I32")
  case object U64  extends SafetensorsDtype("U64")
  case object I64  extends SafetensorsDtype("I64")

  val all: Set[SafetensorsDtype] =
    Set(F16, F32, F64, BF16, U8, I8, U16, I16, U32, I32, U64, I64)

  /** Number of bytes per element for each dtype. */
  def bytesPerElement(dtype: SafetensorsDtype): Int = dtype match {
    case U8 | I8                => 1
    case F16 | BF16 | U16 | I16 => 2
    case F32 | U32 | I32        => 4
    case F64 | U64 | I64        => 8
  }

  def fromString(s: String): Either[String, SafetensorsDtype] =
    all
      .find(_.name == s)
      .toRight(s"Unknown safetensors dtype: '$s'. Valid values: ${all.map(_.name).mkString(", ")}")

  def fromStringUnsafe(s: String): SafetensorsDtype =
    fromString(s).fold(msg => throw new IllegalArgumentException(msg), identity)

}
