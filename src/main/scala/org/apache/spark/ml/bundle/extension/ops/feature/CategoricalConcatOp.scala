package org.apache.spark.ml.bundle.extension.ops.feature

import com.tfs.flashml.core.featuregeneration.transformer.CategoricalColumnsTransformer
import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._

/**
  * Serializer for categorical concat transformer
  */
class CategoricalConcatOp extends SimpleSparkOp[CategoricalColumnsTransformer] {
  override val Model: OpModel[SparkBundleContext, CategoricalColumnsTransformer] = new OpModel[SparkBundleContext,CategoricalColumnsTransformer] {
    override val klazz: Class[CategoricalColumnsTransformer] = classOf[CategoricalColumnsTransformer]

    override def opName: String = "cat_concat"

    override def store(model: Model, obj: CategoricalColumnsTransformer)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      val inputShapes = obj.getInputCols.map(i => sparkToMleapDataShape(dataset.schema(i), dataset): DataShape)

      model.withValue("input_shapes", Value.dataShapeList(inputShapes))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): CategoricalColumnsTransformer = { new CategoricalColumnsTransformer(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: CategoricalColumnsTransformer): CategoricalColumnsTransformer = {
    new CategoricalColumnsTransformer(uid = uid)
  }

  override def sparkInputs(obj: CategoricalColumnsTransformer): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCols)
  }

  override def sparkOutputs(obj: CategoricalColumnsTransformer): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}