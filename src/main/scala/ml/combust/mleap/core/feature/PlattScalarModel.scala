package ml.combust.mleap.core.feature

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.classification.{ClassificationModel, ProbabilisticClassificationModel}
import ml.combust.mleap.core.types._
import ml.combust.mleap.tensor.{DenseTensor, Tensor}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import scala.collection.mutable
import org.apache.spark.ml.linalg.mleap.BLAS


/** Class for multinomial one vs rest models.
  *
  * One vs rest models are comprised of a series of
  * [[ProbabilisticClassificationModel]]s which are used to
  * predict each class.
  *
  * @param classifiers binary classification models
  */
case class PlattScalarModel(coefficients: Array[Vector],
                            intercepts: Array[Double],
                            numClasses: Int) extends Model {
  /** Alias for [[ml.combust.mleap.core.classification.OneVsRestModel#predict]].
    *
    * @param features feature vector
    * @return prediction
    */

  private val score: (Vector,Int) => Double = (features,index) =>
  {
    val m = BLAS.dot(features, coefficients(index)) + intercepts(index)
    1.0 / (1.0 + math.exp(-m))
  }

  def apply(features: Vector): Double = predictOutput(features)

  def predictOutput(features: Vector): Double = {
    predict(features)._2
  }

  def predictProbVector(features:Vector):Vector = {
    predict(features)._1
  }

  /** Predict the class and probability for a feature vector.
    *
    * @param features feature vector
    * @return (predicted class, rawPredictions of class)
    */
  def predict(features: Vector): (Vector,Double) = {
    val probArray = mutable.ArrayBuilder.make[Double]
    val indices = mutable.ArrayBuilder.make[Int]
    var cur = 0
    val (prediction,probability) =
        (0 until numClasses).map { i =>
        val prob = score(Vectors.dense(Array(features(i))), i)
        indices += cur
        probArray += prob
        cur += 1
        (i.toDouble, prob)
      }.maxBy(_._2)

    //(Vectors.sparse(cur, indices.result(), probArray.result()).compressed,prediction)
    (Vectors.dense(probArray.result()),prediction)
  }

  override def inputSchema: StructType = StructType("rawPrediction" -> TensorType.Double(numClasses)).get

  override def outputSchema: StructType = StructType("probability" -> TensorType.Double(numClasses)).get

}