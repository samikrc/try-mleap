package org.apache.spark.ml.bundle.extension.ops.feature

import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import ml.combust.mleap.core.types
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.TopKIntents
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

import scala.util.Try

/**
  * Serializer for gram assembler to make it running in mleap platform.
  */
class TopKIntentsOp extends SimpleSparkOp[TopKIntents] {
  override val Model: OpModel[SparkBundleContext, TopKIntents] = new OpModel[SparkBundleContext, TopKIntents] {
    override val klazz: Class[TopKIntents] = classOf[TopKIntents]

    override def opName: String = "topK"

    override def store(model: Model, obj: TopKIntents)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      val labels = obj.getLabels.toSeq
      val kValue = obj.getKValue

      model.withValue("labels", Value.stringList(labels))
        .withValue("kValue",Value.int(kValue))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): TopKIntents = { new TopKIntents(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: TopKIntents): TopKIntents = {
    new TopKIntents(uid = uid)
  }

  override def sparkInputs(obj: TopKIntents): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: TopKIntents): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}