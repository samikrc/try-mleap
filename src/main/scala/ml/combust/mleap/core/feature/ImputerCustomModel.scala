package ml.combust.mleap.core.feature

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.catalyst.expressions.NaNvl

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
    /*var result = value.get
    value match{
      case Some(a) => result = a
      case None => result = replace
    }
    result*/
    val result = if(value == null || value.toString.equals("NaN")) replace
    else if (inputType != "StringType" && value.isInstanceOf[String]) replace
    else value
    result
  }

  override def inputSchema: StructType = StructType(StructField("input" ,DataType(inputStructType,inputShape))).get

  override def outputSchema: StructType = StructType(StructField("output" ,DataType(inputStructType,inputShape))).get
}