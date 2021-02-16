package org.apache.spark.ml.bundle.extension.ops.feature

import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import ml.combust.mleap.core.types
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.{OneVsRestCustomModel, UpliftTransformer}
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._
import org.apache.spark.ml.classification.{BinaryLogisticRegressionWithDoubleResponseModel, ClassificationModel, LinearSVCModel, LogisticRegressionModel, PlattScalarModel}
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector, VectorUDT, Vectors}
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

      val (baseCoefficients,baseIntercept) = obj.getBaseClassifier match{
        case lr:LogisticRegressionModel => (lr.coefficients,lr.intercept)
        case svm:LinearSVCModel => (svm.coefficients,svm.intercept)
        case _ => throw new Exception("Uplift is supported only for binomial cases")
      }
      val (plattCoefficients,plattIntercept) = if(obj.getBaseClassifier.isInstanceOf[LinearSVCModel] && true)
      (obj.getPlattScaler.models(0).coefficients,obj.getPlattScaler.models(0).intercept) else (Vectors.dense(Array(0.0)),0.0)
      var resultModel = model.withValue("baseCoefficients",Value.vector(baseCoefficients.toArray))
        .withValue("baseIntercept",Value.double(baseIntercept))
      if(obj.getBaseClassifier.isInstanceOf[LinearSVCModel] && true)
        resultModel = resultModel.withValue("plattCoefficients",Value.vector(plattCoefficients.toArray))
        .withValue("plattIntercept",Value.double(plattIntercept))
      resultModel
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): UpliftTransformer = { new UpliftTransformer(uid = "") }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: UpliftTransformer): UpliftTransformer = {
    new UpliftTransformer(uid = uid)
  }

  override def sparkInputs(obj: UpliftTransformer): Seq[ParamSpec] = {
    Seq("features" -> obj.featuresCol)
  }

  override def sparkOutputs(obj: UpliftTransformer): Seq[SimpleParamSpec] = {
    Seq("probability" -> obj.probabilityCol)
  }
}

