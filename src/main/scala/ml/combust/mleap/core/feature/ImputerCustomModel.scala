package ml.combust.mleap.core.feature

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.types._

/**
  * Imputer core logic
  */
case class ImputerCustomModel(inputType:String,surrogateValue:String,inputShape: DataShape) extends Model {

  val inputStructType = inputType match{
    case "StringType" => BasicType.String
    case "DoubleType" => BasicType.Double
    case "IntegerType" => BasicType.Int
    case "FloatType" => BasicType.Float
    case "LongType" => BasicType.Long
  }

  val replace = inputType match {
    case "StringType" => surrogateValue
    case "DoubleType" => surrogateValue.toDouble
    case "IntegerType" => surrogateValue.toInt
    case "FloatType" => surrogateValue.toFloat
    case "LongType" => surrogateValue.toLong
  }

  def apply(value: Any): Any = {
    /*value match {
      case Some(a) => if (inputType != "StringType" && a.isInstanceOf[String]) replace else value.get
      case None => surrogateValue
    }*/
    /*Option(value) match {
      case Some(a) => value
      case None => surrogateValue
    }*/
    value

    /*val temp1 = value.map{ x => val temp = Option(x)
    temp match {
      case Some(a) => if (inputType != "StringType" && a.isInstanceOf[String]) replace else x
      case None => replace
    }}*/
    /*val colValue = Option(value.head)
    colValue match {
      case Some(a) => if (inputType != "StringType" && a.isInstanceOf[String]) replace else colValue.get
      case None => replace
    }*/
  }

  override def inputSchema: StructType = StructType(StructField("input" ,DataType(inputStructType,inputShape))).get

  override def outputSchema: StructType = StructType(StructField("output" ,DataType(inputStructType,inputShape))).get
}