package org.apache.spark.ml.bundle.extension.ops.feature


import ml.bundle.DataShape
import ml.combust.bundle.BundleContext
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.bundle.dsl._
import ml.combust.mleap.core.types
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.ImputerCustom
import org.apache.spark.sql.mleap.TypeConvertersCustom._
import ml.combust.mleap.runtime.types.BundleTypeConverters._
import org.apache.spark.ml.linalg.{Matrix, MatrixUDT, Vector, VectorUDT}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

import scala.util.Try

/**
  * Serializer for gram assembler to make it running in mleap platform.
  */
class ImputerCustomOp extends SimpleSparkOp[ImputerCustom] {
  override val Model: OpModel[SparkBundleContext, ImputerCustom] = new OpModel[SparkBundleContext, ImputerCustom] {
    override val klazz: Class[ImputerCustom] = classOf[ImputerCustom]

    override def opName: String = "imputercustom"

    override def store(model: Model, obj: ImputerCustom)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      val inputSchema = dataset.schema(obj.getInputCol).dataType.toString
      //val inputShapes = obj.getInputCol.map(i => sparkToMleapDataShape(dataset.schema(i),dataset): DataShape)
      val inputShape = sparkToMleapDataShape(dataset.schema(obj.getInputCol),dataset):DataShape

      model.withValue("input_shape", Value.dataShape(inputShape))
        .withValue("input_type", Value.string(inputSchema)).
        withValue("surrogate_value", Value.string(obj.getReplacementValue))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): ImputerCustom = {
      val surrogateValue = model.value("surrogate_value").getString

      new ImputerCustom(uid = "").setReplacementValue(surrogateValue)
    }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: ImputerCustom): ImputerCustom = {
    new ImputerCustom(uid = uid).setReplacementValue(model.getReplacementValue)
  }

  override def sparkInputs(obj: ImputerCustom): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: ImputerCustom): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}
/*import ml.combust.bundle.BundleContext
import ml.bundle.DataShape
import ml.combust.bundle.dsl._
import ml.combust.bundle.op.{OpModel, OpNode}
import ml.combust.mleap.core.feature.ImputerCustomModel
//import ml.combust.mleap.core.types.DataShape
import org.apache.spark.ml.bundle._
import org.apache.spark.ml.feature.ImputerCustom
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.mleap.TypeConvertersCustom._

/**
  * Serialization of spark custom transformer to excute it in the mleap platform. This class serializes the imputer
  * which replace the null values with the given value
  */
class ImputerCustomOp extends SimpleSparkOp[ImputerCustom] {
  override val Model: OpModel[SparkBundleContext, ImputerCustom] = new OpModel[SparkBundleContext, ImputerCustom] {
    override val klazz: Class[ImputerCustom] = classOf[ImputerCustom]

    override def opName: String = "imputercustom"

    override def store(model: Model, obj: ImputerCustom)(implicit context: BundleContext[SparkBundleContext]): Model = {
      assert(context.context.dataset.isDefined, BundleHelper.sampleDataframeMessage(klazz))

      val dataset = context.context.dataset.get
      val inputSchema = dataset.schema(obj.getInputCol).dataType.toString
      val inputShape = sparkToMleapDataShape(dataset.schema(obj.getInputCol),dataset):DataShape

      model.withValue("input_type", Value.string(inputSchema)).
        withValue("surrogate_value", Value.string(obj.replaceValue.toString()))
        .withValue("input_shape",Value.dataShape(inputShape))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): ImputerCustom = {

      val surrogateValue = model.value("surrogate_value").getString

      new ImputerCustom(uid = "").setReplacementValue(surrogateValue)
    }
  }

  override def sparkLoad(uid: String, shape: NodeShape, model: ImputerCustom): ImputerCustom = {
    new ImputerCustom(uid = uid).setReplacementValue(model.getReplacementValue)
  }

  override def sparkInputs(obj: ImputerCustom): Seq[ParamSpec] = {
    Seq("input" -> obj.inputCol)
  }

  override def sparkOutputs(obj: ImputerCustom): Seq[SimpleParamSpec] = {
    Seq("output" -> obj.outputCol)
  }
}*/

/*class ImputerCustomOp extends OpNode[SparkBundleContext, ImputerCustom, ImputerCustomModel] {
  override val Model: OpModel[SparkBundleContext, ImputerCustomModel] = new OpModel[SparkBundleContext, ImputerCustomModel] {
    // the class of the model is needed for when we go to serialize JVM objects
    override val klazz: Class[ImputerCustomModel] = classOf[ImputerCustomModel]

    // a unique name for our op: "string_map"
    // this should be the same as for the MLeap transformer serialization
    override def opName: String = Bundle.BuiltinOps.feature.string_map

    override def store(model: Model, obj: ImputerCustomModel)
                      (implicit context: BundleContext[SparkBundleContext]): Model = {
      // unzip our label map so we can store the label and the value
      // as two parallel arrays, we do this because MLeap Bundles do
      // not support storing data as a map
      val inputType = obj.inputType
      val surrogateValue = obj.surrogateValue

      // add the labels and values to the Bundle model that
      // will be serialized to our MLeap bundle
      model.withValue("inputType", Value.string(inputType)).
        withValue("values", Value.string(surrogateValue))
    }

    override def load(model: Model)
                     (implicit context: BundleContext[SparkBundleContext]): ImputerCustomModel = {
      // retrieve our list of labels
      val labels = model.value("inputType").getString

      // retrieve our list of values
      val values = model.value("values").getString

      // reconstruct the model using the parallel labels and values
      ImputerCustomModel(labels,values)
    }
  }
  override val klazz: Class[ImputerCustom] = classOf[ImputerCustom]

  override def name(node: ImputerCustom): String = node.uid

  override def model(node: ImputerCustom): ImputerCustomModel = new ImputerCustomModel("","")

  override def load(node: Node, model: ImputerCustomModel)
                   (implicit context: BundleContext[SparkBundleContext]): ImputerCustom = {
    new ImputerCustom(uid = node.name).
      setInputCol(node.shape.standardInput.name)
  }

  override def shape(node: ImputerCustom)(implicit context: BundleContext[SparkBundleContext]): NodeShape =
    NodeShape().withStandardIO(node.getInputCol, node.getOutputCol)
}*/
