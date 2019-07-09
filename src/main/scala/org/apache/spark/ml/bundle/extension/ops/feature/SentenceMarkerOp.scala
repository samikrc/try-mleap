package org.apache.spark.ml.bundle.extension.ops.feature

import ml.combust.bundle.{BundleContext}
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.mleap.core.feature.CaseNormalizationModel
import ml.combust.mleap.runtime.MleapContext
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.feature.SentenceMarker

/**
  * Created by hollinwilkins on 2/5/17.
  */
import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.OpModel
import ml.combust.mleap.bundle.ops.MleapOp
import ml.combust.mleap.core.feature.RegexReplaceModel
import ml.combust.mleap.runtime.MleapContext
import ml.combust.mleap.runtime.transformer.feature.RegexReplaceCustom
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.SentenceMarker
import org.apache.spark.sql.mleap.TypeConvertersCustom.sparkToMleapDataShape

class SentenceMarkerOp extends SimpleSparkOp[SentenceMarker] {
  override val Model: OpModel[SparkBundleContext, SentenceMarker] = new OpModel[SparkBundleContext,SentenceMarker] {
    override val klazz: Class[SentenceMarker] = classOf[SentenceMarker]

    override def opName: String = "sentence_marker"

    override def store(model: Model, obj: SentenceMarker)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      // add the labels and values to the Bundle model that
      // will be serialized to our MLeap bundle
      model
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): SentenceMarker = { new SentenceMarker(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: SentenceMarker): SentenceMarker = {
    new SentenceMarker(uid = uid)
  }

  override def sparkInputs(obj: SentenceMarker): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: SentenceMarker): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}
