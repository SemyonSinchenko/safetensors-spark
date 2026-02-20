package io.github.semyonsinchenko.safetensors.expressions

import io.github.semyonsinchenko.safetensors.core.{SafetensorsDtype, TensorSchema}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionInfo, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import java.nio.{ByteBuffer, ByteOrder}

/** Catalyst expression: st_to_array(tensorCol)
  *
  * Input: Tensor Struct (data: BinaryType, shape: ArrayType(IntegerType), dtype: StringType)
  * Output: ArrayType(FloatType) — flat, row-major
  *
  * Parses the `data` bytes according to the `dtype` field and upcasts all values to Float32.
  *
  * WARNING — BF16/F16 read is a lossy upcast: For BF16, the upcast to Float32 is lossless (BF16 is
  * a strict subset of Float32's exponent range). For F16, values outside Float32 precision range
  * are preserved. However, downstream Spark operations lose the original 16-bit type information.
  * Use the raw `data` field for lossless round-trips.
  *
  * NOTE on BF16: BF16 is not present in the official JSON schema dtype pattern
  * ([UIF])(8|16|32|64|128|256). It is accepted here as a special case. See §1.1 of AGENTS.md.
  *
  * Extends UnaryExpression with CodegenFallback (interpreted execution).
  */
case class StToArrayExpression(child: Expression) extends UnaryExpression with CodegenFallback {

  override def dataType: DataType = ArrayType(FloatType, containsNull = false)

  override def nullable: Boolean = child.nullable

  override def checkInputDataTypes(): TypeCheckResult = child.dataType match {
    case st: StructType if TensorSchema.isTensorStruct(st) =>
      TypeCheckResult.TypeCheckSuccess
    case other =>
      TypeCheckResult.TypeCheckFailure(
        s"st_to_array: argument must be a Tensor Struct " +
          s"(StructType with data/shape/dtype fields), got ${other.simpleString}"
      )
  }

  override def nullSafeEval(input: Any): Any = {
    val row       = input.asInstanceOf[InternalRow]
    val dataBytes = row.getBinary(0)
    val dtypeStr  = row.getUTF8String(2).toString

    val dtype  = SafetensorsDtype.fromStringUnsafe(dtypeStr)
    val floats = decodeToFloats(dataBytes, dtype)
    // Box primitive Array[Float] before passing to GenericArrayData.
    // Array[Float] (primitive) cannot be cast directly to Array[Any].
    new GenericArrayData(floats.map(f => f.asInstanceOf[AnyRef]))
  }

  private def decodeToFloats(bytes: Array[Byte], dtype: SafetensorsDtype): Array[Float] = {
    val buf       = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val bytesEach = SafetensorsDtype.bytesPerElement(dtype)
    val nElem     = if (bytesEach > 0) bytes.length / bytesEach else 0
    val result    = new Array[Float](nElem)

    for (i <- 0 until nElem)
      result(i) = dtype match {
        case SafetensorsDtype.F32 => buf.getFloat()
        case SafetensorsDtype.F64 => buf.getDouble().toFloat
        case SafetensorsDtype.I8  => buf.get().toFloat
        case SafetensorsDtype.U8  => (buf.get() & 0xff).toFloat
        case SafetensorsDtype.I16 => buf.getShort().toFloat
        case SafetensorsDtype.U16 => (buf.getShort() & 0xffff).toFloat
        case SafetensorsDtype.I32 => buf.getInt().toFloat
        case SafetensorsDtype.U32 => (buf.getInt() & 0xffffffffL).toFloat
        case SafetensorsDtype.I64 => buf.getLong().toFloat
        // U64: convert via bit manipulation to avoid string-parse overflow.
        // If the high bit is 0 (value fits in signed Long), cast directly.
        // If the high bit is 1, right-shift by 1 (halve, losing LSB) then
        // multiply by 2.0f. This preserves the approximate magnitude without
        // a string intermediary that would throw for values > Long.MAX_VALUE.
        case SafetensorsDtype.U64 =>
          val raw = buf.getLong()
          if (raw >= 0L) raw.toFloat
          else ((raw >>> 1).toFloat) * 2.0f
        // BF16: lossless upcast to Float32 by zero-extending the 16-bit value
        // into the top 16 bits of a Float32 bit pattern.
        // NOTE: BF16 is a special case outside the official JSON schema — see §1.1.
        case SafetensorsDtype.BF16 =>
          val bits = (buf.getShort() & 0xffff) << 16
          java.lang.Float.intBitsToFloat(bits)
        // F16: upcast IEEE 754 float16 to Float32
        case SafetensorsDtype.F16 =>
          float16ToFloat(buf.getShort())
      }

    result
  }

  /** Convert an IEEE 754 Float16 (as a Short) to Float32. */
  private def float16ToFloat(bits16: Short): Float = {
    val b      = bits16 & 0xffff
    val sign   = (b >>> 15) & 0x1
    val exp16  = (b >>> 10) & 0x1f
    val mant16 = b & 0x3ff

    val (exp32, mant32) = if (exp16 == 0x1f) {
      (0xff, mant16 << 13) // Inf or NaN
    } else if (exp16 == 0) {
      if (mant16 == 0) (0, 0) // zero
      else {
        // subnormal f16 -> normalised f32
        var m = mant16
        var e = 1
        while ((m & 0x400) == 0) { m <<= 1; e += 1 }
        (127 - 15 - e + 1, (m & 0x3ff) << 13)
      }
    } else {
      (exp16 - 15 + 127, mant16 << 13)
    }

    val bits32 = (sign << 31) | (exp32 << 23) | mant32
    java.lang.Float.intBitsToFloat(bits32)
  }

  override protected def withNewChildInternal(newChild: Expression): Expression =
    copy(child = newChild)

}

object StToArrayExpression {

  val functionDescription: (String, ExpressionInfo, Seq[Expression] => Expression) = (
    "st_to_array",
    new ExpressionInfo(
      classOf[StToArrayExpression].getName,
      null,
      "st_to_array",
      "(tensorCol) - Decodes a Tensor Struct's raw bytes to a flat ArrayType(FloatType). " +
        "All dtypes are upcast to Float32.",
      "",
      "",
      "",
      "",
      "",
      "",
      "misc_funcs"
    ),
    (children: Seq[Expression]) => {
      require(
        children.length == 1,
        s"st_to_array requires exactly 1 argument, got ${children.length}"
      )
      StToArrayExpression(children(0))
    }
  )

}
