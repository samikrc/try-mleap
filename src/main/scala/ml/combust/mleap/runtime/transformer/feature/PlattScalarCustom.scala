package ml.combust.mleap.runtime.transformer.feature

import ml.combust.mleap.core.feature.PlattScalarModel
import ml.combust.mleap.core.types._
import ml.combust.mleap.runtime.function.{StructSelector, UserDefinedFunction}
import ml.combust.mleap.tensor.Tensor
import ml.combust.mleap.core.util.VectorConverters._
import ml.combust.mleap.runtime.frame.{FrameBuilder, MultiTransformer, Row, Transformer}
import org.apache.spark.ml.linalg.Vectors

import scala.util.Try

/**
  * Created by hwilkins on 10/22/15.
  */
case class PlattScalarCustom(override val uid: String = Transformer.uniqueName("platt_scalar"),
                           override val shape: NodeShape,
                           override val model: PlattScalarModel) extends MultiTransformer {
  override val exec: UserDefinedFunction = {
    val f = shape.getOutput("probability") match {
      case Some(_) =>
        (features: Tensor[Double]) => {
          val probability:Tensor[Double] = model.predictProbVector(features)
          Row(probability)
        }
    }
    UserDefinedFunction(f, outputSchema, inputSchema)
  }
}
