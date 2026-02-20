package io.github.semyonsinchenko.safetensors.core

import org.apache.spark.sql.types._

/**
 * The canonical Tensor Struct type used for all read, write, and Catalyst
 * expression operations in this connector.
 *
 * Schema:
 *   StructType(
 *     StructField("data",  BinaryType,            nullable = false),
 *     StructField("shape", ArrayType(IntegerType), nullable = false),
 *     StructField("dtype", StringType,            nullable = false)
 *   )
 *
 * - data:  raw little-endian bytes of the tensor, exactly as stored in the
 *          safetensors byte buffer.
 * - shape: dimension sizes (e.g. [3, 224, 224]). Empty array [] for scalars.
 * - dtype: one of the SafetensorsDtype string representations.
 *
 * Rationale: BinaryType is the only Spark type that can losslessly represent
 * all safetensors dtypes including BF16 and F16, for which Spark has no native
 * numeric type.
 */
object TensorSchema {

  val DATA_FIELD  = "data"
  val SHAPE_FIELD = "shape"
  val DTYPE_FIELD = "dtype"

  /** The canonical Tensor Struct StructType. */
  val schema: StructType = StructType(Seq(
    StructField(DATA_FIELD,  BinaryType,                        nullable = false),
    StructField(SHAPE_FIELD, ArrayType(IntegerType, false), nullable = false),
    StructField(DTYPE_FIELD, StringType,                        nullable = false),
  ))

  /**
   * Returns true if the given StructType matches the canonical Tensor Struct
   * (field names and types must match exactly; nullability is not checked).
   */
  def isTensorStruct(st: StructType): Boolean =
    st.fields.length == 3 &&
    st.fieldNames.toSeq == Seq(DATA_FIELD, SHAPE_FIELD, DTYPE_FIELD) &&
    st(DATA_FIELD).dataType  == BinaryType &&
    st(SHAPE_FIELD).dataType == ArrayType(IntegerType, false) &&
    st(DTYPE_FIELD).dataType == StringType

  /** Numeric Spark element types accepted for ArrayType write inputs. */
  val numericElementTypes: Set[DataType] =
    Set(FloatType, DoubleType, IntegerType, LongType, ShortType, ByteType)

  /**
   * Returns true if the given DataType is a numeric ArrayType that the
   * connector can encode into tensor bytes.
   */
  def isNumericArrayType(dt: DataType): Boolean = dt match {
    case ArrayType(elementType, _) => numericElementTypes.contains(elementType)
    case _                         => false
  }
}
