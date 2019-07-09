package ml.combust.mleap.bundle.ops.feature

import ml.combust.bundle.BundleContext
import ml.combust.mleap.core.feature.UpliftModel
import ml.combust.mleap.runtime.transformer.feature.Uplift
import ml.combust.bundle.op.OpModel
import ml.combust.bundle.dsl._
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.types.BundleTypeConverters._

/**
  * Serializer for Gram assembler to run in the mleap platform
  */
class UpliftOp extends MleapOp[Uplift, UpliftModel] {
  override val Model: OpModel[MleapContext, UpliftModel] = new OpModel[MleapContext, UpliftModel] {
    override val klazz: Class[UpliftModel] = classOf[UpliftModel]

    override def opName: String = "uplift"

    override def store(model: Model, obj: UpliftModel)
                      (implicit context: BundleContext[MleapContext]): Model = {
      model.withValue("input_shape", Value.dataShape(mleapToBundleShape(obj.inputShape)))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[MleapContext]): UpliftModel = {
      val inputShape = bundleToMleapShape(model.value("input_shape").getDataShape)
      UpliftModel(inputShape)
    }
  }

  override def model(node: Uplift): UpliftModel = node.model
}
