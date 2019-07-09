package ml.combust.mleap.bundle.ops.feature

import ml.combust.bundle.BundleContext
import ml.combust.mleap.core.feature.CatModel
import ml.combust.mleap.runtime.transformer.feature.CatConcat
import ml.combust.bundle.op.OpModel
import ml.combust.bundle.dsl._
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.types.BundleTypeConverters._

/**
  * Created by hollinwilkins on 8/22/16.
  */
class CatOp extends MleapOp[CatConcat, CatModel] {
  override val Model: OpModel[MleapContext, CatModel] = new OpModel[MleapContext, CatModel] {
    override val klazz: Class[CatModel] = classOf[CatModel]

    override def opName: String = "cat_concat"

    override def store(model: Model, obj: CatModel)
                      (implicit context: BundleContext[MleapContext]): Model = {
      model.withValue("input_shapes", Value.dataShapeList(obj.inputShapes.map(mleapToBundleShape)))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[MleapContext]): CatModel = {
      val inputShapes = model.value("input_shapes").getDataShapeList.map(bundleToMleapShape)
      CatModel(inputShapes)
    }
  }

  override def model(node: CatConcat): CatModel = node.model
}