package org.apache.spark.ml.bundle.extension.ops.feature

import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.{RegexReplacementTransformer, VectorAssembler}

/**
  * Serialization of spark custom transformer to excute it in the mleap platform. This class serializes the regex replace transformer which
  * replaces regex pattern with the given replacement
  */
class RegexReplaceOp extends SimpleSparkOp[RegexReplacementTransformer] {
  override val Model: OpModel[SparkBundleContext, RegexReplacementTransformer] = new OpModel[SparkBundleContext,RegexReplacementTransformer] {
    override val klazz: Class[RegexReplacementTransformer] = classOf[RegexReplacementTransformer]

    override def opName: String = "regex_replace"

    override def store(model: Model, obj: RegexReplacementTransformer)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val (labels, values) = obj.getRegexReplacements.toSeq.unzip

      // add the key and values to the Bundle model that
      // will be serialized to our MLeap bundle
      model.withValue("labels", Value.stringList(labels)).
        withValue("values", Value.stringList(values))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): RegexReplacementTransformer = { new RegexReplacementTransformer(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: RegexReplacementTransformer): RegexReplacementTransformer = {
    new RegexReplacementTransformer(uid = uid)
  }

  override def sparkInputs(obj: RegexReplacementTransformer): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: RegexReplacementTransformer): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}