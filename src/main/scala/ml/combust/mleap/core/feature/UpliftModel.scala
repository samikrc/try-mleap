package ml.combust.mleap.core.feature

import com.github.aztek.porterstemmer.PorterStemmer
import ml.combust.mleap.core.{Model, types}
import ml.combust.mleap.core.types._
import ml.combust.mleap.tensor.Tensor

import scala.collection.mutable

/**
  * Core logic for stop word remover
  */
case class UpliftModel(inputShape: DataShape) extends Model {
  def apply(predictions: Seq[Any]): Array[Double] = {
    val values = mutable.ArrayBuilder.make[Double]
    predictions.foreach {
      case x:Tensor[Double] =>
        val scores = for (score <- x.toArray) yield score
        scores.foreach(x => values += x.toDouble)
    }
    values.result()
  }

  override def inputSchema: StructType = {
    val inputField = StructField(s"input", DataType(BasicType.Double, inputShape))
    StructType(inputField).get
  }
  override def outputSchema: StructType = StructType(StructField("output" -> ListType(BasicType.Double))).get
}
