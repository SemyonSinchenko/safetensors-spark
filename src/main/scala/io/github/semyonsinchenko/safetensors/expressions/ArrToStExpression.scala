package io.github.semyonsinchenko.safetensors.expressions

import io.github.semyonsinchenko.safetensors.core.{SafetensorsDtype, TensorSchema}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionInfo}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import java.nio.{ByteBuffer, ByteOrder}

/**
 * Catalyst expression: arr_to_st(arrayCol, shape, dtype)
 *
 * Input:
 *   arrayCol : ArrayType(FloatType) — flat float array
 *   shape    : ArrayType(IntegerType) — target tensor dimensions
 *   dtype    : StringType (literal) — safetensors dtype string
 *
 * Output: Tensor Struct (data: BinaryType, shape: ArrayType(IntegerType), dtype: StringType)
 *
 * Converts a Spark float array into a Tensor Struct by encoding the Float32
 * values as raw bytes in the specified dtype. Byte order is little-endian.
 *
 * WARNING — BF16/F16 conversion is approximate:
 *   When dtype = "BF16", Float32 values are converted using simple truncation
 *   (top 16 bits of the IEEE 754 Float32 bit pattern), NOT round-to-nearest-even.
 *   This is fast but may differ from PyTorch/NumPy by up to 1 ULP in the BF16
 *   mantissa. BF16 is a special case not present in the official JSON schema —
 *   see §1.1 of AGENTS.md for details.
 *   The same truncation applies to F16 (Float32 → IEEE 754 float16).
 *
 * Extends Expression with CodegenFallback (interpreted execution).
 */
case class ArrToStExpression(
  arrayCol: Expression,
  shape:    Expression,
  dtype:    Expression,
) extends Expression with CodegenFallback {

  override def children: Seq[Expression] = Seq(arrayCol, shape, dtype)

  override def dataType: DataType = TensorSchema.schema

  override def nullable: Boolean = false

  override def checkInputDataTypes(): TypeCheckResult = {
    val arrayOk = arrayCol.dataType match {
      case ArrayType(FloatType, _) => true
      case _                       => false
    }
    val shapeOk = shape.dataType match {
      case ArrayType(IntegerType, _) => true
      case _                         => false
    }
    val dtypeOk = dtype.dataType == StringType

    if (!arrayOk)
      TypeCheckResult.TypeCheckFailure(
        s"arr_to_st: first argument must be ArrayType(FloatType), got ${arrayCol.dataType.simpleString}")
    else if (!shapeOk)
      TypeCheckResult.TypeCheckFailure(
        s"arr_to_st: second argument must be ArrayType(IntegerType), got ${shape.dataType.simpleString}")
    else if (!dtypeOk)
      TypeCheckResult.TypeCheckFailure(
        s"arr_to_st: third argument must be StringType, got ${dtype.dataType.simpleString}")
    else
      TypeCheckResult.TypeCheckSuccess
  }

  override def eval(input: InternalRow): Any = {
    val arrayData = arrayCol.eval(input)
    val shapeData = shape.eval(input)
    val dtypeData = dtype.eval(input)

    if (arrayData == null || dtypeData == null) return null

    val floats    = arrayData.asInstanceOf[org.apache.spark.sql.catalyst.util.ArrayData]
    val shapeArr  = shapeData.asInstanceOf[org.apache.spark.sql.catalyst.util.ArrayData]
    val dtypeStr  = dtypeData.asInstanceOf[UTF8String].toString

    val safeDtype = SafetensorsDtype.fromStringUnsafe(dtypeStr)
    val nElem     = floats.numElements()
    val bytes     = encodeFloats(floats, nElem, safeDtype)

    val row = new GenericInternalRow(3)
    row.update(0, bytes)                                     // data
    row.update(1, shapeArr)                                  // shape (pass through)
    row.update(2, UTF8String.fromString(safeDtype.name))     // dtype
    row
  }

  private def encodeFloats(
    floats:   org.apache.spark.sql.catalyst.util.ArrayData,
    nElem:    Int,
    dtype:    SafetensorsDtype,
  ): Array[Byte] = {
    val bytesPerElem = SafetensorsDtype.bytesPerElement(dtype)
    val buf = ByteBuffer.allocate(nElem * bytesPerElem)
    buf.order(ByteOrder.LITTLE_ENDIAN)

    for (i <- 0 until nElem) {
      val f = floats.getFloat(i)
      dtype match {
        case SafetensorsDtype.F32  => buf.putFloat(f)
        case SafetensorsDtype.F64  => buf.putDouble(f.toDouble)
        case SafetensorsDtype.I8   => buf.put(f.toByte)
        case SafetensorsDtype.U8   => buf.put((f.toInt & 0xFF).toByte)
        case SafetensorsDtype.I16  => buf.putShort(f.toShort)
        case SafetensorsDtype.U16  => buf.putShort((f.toInt & 0xFFFF).toShort)
        case SafetensorsDtype.I32  => buf.putInt(f.toInt)
        case SafetensorsDtype.U32  => buf.putInt(f.toLong.toInt)
        case SafetensorsDtype.I64  => buf.putLong(f.toLong)
        case SafetensorsDtype.U64  => buf.putLong(f.toLong)
        // BF16: truncation of top 16 bits of Float32 IEEE 754 bit pattern.
        // NOTE: BF16 is not in the JSON schema regex — see §1.1.
        // This is approximate (not round-to-nearest-even). See class Scaladoc.
        case SafetensorsDtype.BF16 =>
          val bits = java.lang.Float.floatToRawIntBits(f)
          buf.putShort((bits >>> 16).toShort)
        // F16: truncate Float32 mantissa to 10 bits (approximate).
        // NOTE: This is a lossy truncation, not round-to-nearest-even.
        case SafetensorsDtype.F16  =>
          buf.putShort(floatToFloat16Truncate(f))
      }
    }

    buf.array()
  }

  /**
   * Approximate Float32 → Float16 conversion via truncation (not RNE).
   * WARNING: See class Scaladoc for limitations.
   */
  private def floatToFloat16Truncate(f: Float): Short = {
    val bits    = java.lang.Float.floatToRawIntBits(f)
    val sign    = (bits >>> 31) & 0x1
    val exp32   = (bits >>> 23) & 0xFF
    val mant32  = bits & 0x7FFFFF

    if (exp32 == 0xFF) {
      // Inf or NaN
      val f16 = (sign << 15) | 0x7C00 | (if (mant32 != 0) 0x200 else 0)
      f16.toShort
    } else if (exp32 == 0) {
      // Subnormal or zero -> zero in F16
      (sign << 15).toShort
    } else {
      val exp16 = exp32 - 127 + 15
      if (exp16 >= 0x1F) {
        // Overflow -> Inf
        ((sign << 15) | 0x7C00).toShort
      } else if (exp16 <= 0) {
        // Underflow -> zero
        (sign << 15).toShort
      } else {
        val mant16 = mant32 >>> 13  // truncate (not round)
        ((sign << 15) | (exp16 << 10) | mant16).toShort
      }
    }
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression =
    copy(arrayCol = newChildren(0), shape = newChildren(1), dtype = newChildren(2))
}

object ArrToStExpression {
  val functionDescription: (String, ExpressionInfo, Seq[Expression] => Expression) = (
    "arr_to_st",
    new ExpressionInfo(
      classOf[ArrToStExpression].getName,
      null,
      "arr_to_st",
      "(arrayCol, shape, dtype) - Converts a flat FloatType array into a Tensor Struct " +
        "by encoding values as raw bytes in the specified safetensors dtype.",
      "",
      "",
      "",
      "",
      "",
      "",
      "misc_funcs",
    ),
    (children: Seq[Expression]) => {
      require(children.length == 3,
        s"arr_to_st requires exactly 3 arguments, got ${children.length}")
      ArrToStExpression(children(0), children(1), children(2))
    },
  )
}
