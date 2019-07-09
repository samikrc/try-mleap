package ml.combust.mleap.runtime.transformer.feature

import ml.combust.mleap.core.feature.GramAssemblerModel
import ml.combust.mleap.core.types._
import ml.combust.mleap.tensor.Tensor
import ml.combust.mleap.core.util.VectorConverters._
import ml.combust.mleap.runtime.frame.{FrameBuilder, Row, Transformer}
import ml.combust.mleap.runtime.function.{StructSelector, UserDefinedFunction}

import scala.util.Try

case class GramAssembler(override val uid: String = Transformer.uniqueName("gram_assembler"),
                           override val shape: NodeShape,
                           override val model: GramAssemblerModel) extends Transformer {
  private val f = (values: Row) => model(values.toSeq)
  val exec: UserDefinedFunction = UserDefinedFunction(f,
    outputSchema.fields.head.dataType,
    Seq(SchemaSpec(inputSchema)))

  val outputCol: String = outputSchema.fields.head.name
  val inputCols: Seq[String] = inputSchema.fields.map(_.name)
  private val inputSelector: StructSelector = StructSelector(inputCols)

  override def transform[TB <: FrameBuilder[TB]](builder: TB): Try[TB] = {
    builder.withColumn(outputCol, inputSelector)(exec)
  }
}