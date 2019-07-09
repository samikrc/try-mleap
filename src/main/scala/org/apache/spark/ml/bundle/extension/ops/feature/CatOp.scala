package org.apache.spark.ml.bundle.extension.ops.feature

import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.{CategoricalConcatTransformer, VectorAssembler}
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._

/**
  * Serializer for categorical concat transformer
  */
class CatOp extends SimpleSparkOp[CategoricalConcatTransformer] {
  override val Model: OpModel[SparkBundleContext, CategoricalConcatTransformer] = new OpModel[SparkBundleContext,CategoricalConcatTransformer] {
    override val klazz: Class[CategoricalConcatTransformer] = classOf[CategoricalConcatTransformer]

    override def opName: String = "cat_concat"

    override def store(model: Model, obj: CategoricalConcatTransformer)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      val inputShapes = obj.getInputCols.map(i => sparkToMleapDataShape(dataset.schema(i), dataset): DataShape)

      model.withValue("input_shapes", Value.dataShapeList(inputShapes))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): CategoricalConcatTransformer = { new CategoricalConcatTransformer(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: CategoricalConcatTransformer): CategoricalConcatTransformer = {
    new CategoricalConcatTransformer(uid = uid)
  }

  override def sparkInputs(obj: CategoricalConcatTransformer): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCols)
  }

  override def sparkOutputs(obj: CategoricalConcatTransformer): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}