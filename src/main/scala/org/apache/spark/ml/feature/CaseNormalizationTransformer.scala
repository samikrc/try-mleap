package org.apache.spark.ml.feature

import java.io.Serializable

import ml.combust.mleap.core.feature.CaseNormalizationModel
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

/** Transformer to convert all word to lower case.
  * @author Neelesh Sambhajiche <neelesh.sa@247-inc.com>
  * @since 24/8/18
  */

class CaseNormalizationTransformer(override val uid: String)
  extends UnaryTransformer[String, String, CaseNormalizationTransformer]
    with Serializable
    with DefaultParamsWritable{

  def this() = this(Identifiable.randomUID("case_norm"))

  override protected def createTransformFunc: String => String = {
    _.toLowerCase
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be String type but got $inputType.")
  }

  override protected def outputDataType: DataType = {
    StringType
  }
}

object CaseNormalizationTransformer
  extends DefaultParamsReadable[CaseNormalizationTransformer] {
  override def load(path: String): CaseNormalizationTransformer = super.load(path)
}
