package com.tfs.flashml.core.featuregeneration.transformer

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasHandleInvalid, HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{Column, DataFrame, Dataset}
import org.apache.spark.sql.functions.{array, concat, lit}
import org.apache.spark.sql.types._

/** This first concatenated the column name to the column value and the groups all the categorical
  * columns into an Array
  * @author Neelesh Sambhajiche <neelesh.sa@247-inc.com>
  * @since 22/8/18
  */

class CategoricalColumnsTransformer (override val uid: String)
  extends Transformer with HasInputCols with HasOutputCol
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("cat_concat"))

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def copy(extra: ParamMap): CategoricalColumnsTransformer = defaultCopy(extra)

  def transform(df: Dataset[_]): DataFrame = {
    val nameConcatedColumnArray: Array[Column] = for (column <- $(inputCols)) yield concat(lit(column + "_"), df(column))
    df.withColumn($(outputCol), array(nameConcatedColumnArray: _*))
  }

  override def transformSchema(schema: StructType): StructType = {
    // Check that the input type is a string
    $(inputCols) foreach { strCol: String =>
      val idx = schema.fieldIndex(strCol)
      val field = schema.fields(idx)
      //todo MODIFIED
      if (field.dataType != StringType && field.dataType != IntegerType && field.dataType!=DoubleType) {
        throw new Exception(s"Input type ${field.dataType} did not match required type StringType/IntegerType")
      }
    }
    // Add the return field
    schema.add(StructField($(outputCol), ArrayType(StringType, true), false))
  }
}

object CategoricalColumnsTransformer
  extends DefaultParamsReadable[CategoricalColumnsTransformer] {
  override def load(path: String): CategoricalColumnsTransformer = super.load(path)
}
