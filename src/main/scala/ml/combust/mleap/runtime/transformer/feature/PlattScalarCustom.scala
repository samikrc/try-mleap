package ml.combust.mleap.runtime.transformer.feature

import ml.combust.mleap.core.feature.PlattScalarModel
import ml.combust.mleap.core.types._
import ml.combust.mleap.runtime.function.{StructSelector, UserDefinedFunction}
import ml.combust.mleap.tensor.Tensor
import ml.combust.mleap.core.util.VectorConverters._
import ml.combust.mleap.runtime.frame.{FrameBuilder, MultiTransformer, Row, Transformer}
import org.apache.spark.ml.linalg.Vectors

import scala.util.Try

/**
  * Created by hwilkins on 10/22/15.
  */
case class PlattScalarCustom(override val uid: String = Transformer.uniqueName("platt_scalar"),
                           override val shape: NodeShape,
                           override val model: PlattScalarModel) extends MultiTransformer {
  /*override val exec: UserDefinedFunction = {
    val f = shape.getOutput("probability") match {
      case Some(_) =>
        (features: Tensor[Double]) => {
          val probability:Tensor[Double] = model.predictProbVector(features)
          Row(probability)
        }
    }
    UserDefinedFunction(f, outputSchema, inputSchema)
  }*/
  private val f = (values: Row) => {
    val t = Some(values.head)
    model(values.head):Tensor[Double]}
  val exec: UserDefinedFunction = UserDefinedFunction(f,
    outputSchema.fields.head.dataType,
    Seq(SchemaSpec(inputSchema)))
  private val f1 = (values: Row) => {
    values.head}
  val exec1: UserDefinedFunction = UserDefinedFunction(f1,
    outputSchema.fields.last.dataType,
    Seq(SchemaSpec(StructType("prediction" -> ScalarType.Double).get)))
  private val f2 = (values: Row) => {
    model.predictByMaxProb(values.head)}
  val exec2: UserDefinedFunction = UserDefinedFunction(f2,
    outputSchema.fields.last.dataType,
    Seq(SchemaSpec(inputSchema)))

  val outputCol: String = outputSchema.fields.head.name
  val outputType = outputSchema.fields.head.dataType
  val svmPredOutputType = outputSchema.fields.last.dataType
  val inputCols: Seq[String] = inputSchema.fields.map(_.name)
  val svmPredOutputCol = "svmPrediction"
  private val inputSelector: StructSelector = StructSelector(inputCols)
  private val svmPredSelector:StructSelector = StructSelector(Seq("prediction"))


  override def transform[TB <: FrameBuilder[TB]](builder: TB): Try[TB] = {
    builder
      .withColumn(outputCol, inputSelector)(exec)
      .get
      .withColumn(svmPredOutputCol,svmPredSelector)(exec1)
      .get
      .drop("prediction")
      .get
      .withColumn("prediction",StructSelector(Seq("probability")))(exec2)
  }
}
