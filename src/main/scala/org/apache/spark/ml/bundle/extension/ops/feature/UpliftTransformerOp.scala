package org.apache.spark.ml.bundle.extension.ops.feature

import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import ml.combust.mleap.core.types
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.UpliftTransformer
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

import scala.util.Try

/**
  * Serializer for gram assembler to make it running in mleap platform.
  */
class UpliftTransformerOp extends SimpleSparkOp[UpliftTransformer] {
  override val Model: OpModel[SparkBundleContext, UpliftTransformer] = new OpModel[SparkBundleContext, UpliftTransformer] {
    override val klazz: Class[UpliftTransformer] = classOf[UpliftTransformer]

    override def opName: String = "uplift"

    override def store(model: Model, obj: UpliftTransformer)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      //val inputShapes = obj.getInputCol.map(i => sparkToMleapDataShape(dataset.schema(i),dataset): DataShape)
      val inputShape = sparkToMleapDataShape(dataset.schema(obj.getInputCol),dataset):DataShape

      model.withValue("input_shape", Value.dataShape(inputShape))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): UpliftTransformer = { new UpliftTransformer(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: UpliftTransformer): UpliftTransformer = {
    new UpliftTransformer(uid = uid)
  }

  override def sparkInputs(obj: UpliftTransformer): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: UpliftTransformer): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}