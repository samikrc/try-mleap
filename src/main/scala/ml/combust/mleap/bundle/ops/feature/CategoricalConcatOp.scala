package ml.combust.mleap.bundle.ops.feature

import ml.combust.bundle.BundleContext
import ml.combust.mleap.core.feature.CategoricalModel
import ml.combust.mleap.runtime.transformer.feature.CategoricalConcat
import ml.combust.bundle.op.OpModel
import ml.combust.bundle.dsl._
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.types.BundleTypeConverters._

/**
  * Created by hollinwilkins on 8/22/16.
  */
class CategoricalConcatOp extends MleapOp[CategoricalConcat, CategoricalModel] {
  override val Model: OpModel[MleapContext, CategoricalModel] = new OpModel[MleapContext, CategoricalModel] {
    override val klazz: Class[CategoricalModel] = classOf[CategoricalModel]

    override def opName: String = "cat_concat"

    override def store(model: Model, obj: CategoricalModel)
                      (implicit context: BundleContext[MleapContext]): Model = {
      model.withValue("input_shapes", Value.dataShapeList(obj.inputShapes.map(mleapToBundleShape)))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[MleapContext]): CategoricalModel = {
      val inputShapes = model.value("input_shapes").getDataShapeList.map(bundleToMleapShape)
      CategoricalModel(inputShapes)
    }
  }

  override def model(node: CategoricalConcat): CategoricalModel = node.model
}