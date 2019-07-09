package ml.combust.mleap.core.feature

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.annotation.SparkCode
import ml.combust.mleap.core.types._
import ml.combust.mleap.tensor.{DenseTensor, SparseTensor}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.collection.mutable

/** Class for a vector assembler model.
  *
  * Vector assemblers take an input set of doubles and vectors
  * and create a new vector out of them. This is primarily used
  * to get all desired features into one vector before training
  * a model.
  */
case class CatModel(inputShapes: Seq[DataShape]) extends Model {

  val outputSize: Int = inputShapes.map {
    case ScalarShape(_) => 1
    case _ => throw new IllegalArgumentException("must provide scalar and vector shapes as inputs")
  }.sum

  /** Assemble a feature vector from a set of input features.
    *
    * @param vv all input feature values
    * @return assembled vector
    */
  def apply(valueArray:Seq[Any],nameArray:Seq[String]) : Seq[String] = {
    val values = mutable.ArrayBuilder.make[String]
    nameArray.zip(valueArray).foreach {
      case (a: String, b: String) => values += a + "_" + b
      case _ => ""
    }
    values.result().toSeq
  }

  override def inputSchema: StructType = {
    val inputFields = inputShapes.zipWithIndex.map {
      case (shape, i) => StructField(s"input$i", DataType(BasicType.String, shape))
    }

    StructType(inputFields).get
  }

  override def outputSchema: StructType = StructType(StructField("output" -> ListType(BasicType.String))).get
}
