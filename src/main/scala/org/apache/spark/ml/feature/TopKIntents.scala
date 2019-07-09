package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, _}

/**
  * Calculate the top K intents for Multi Intent models.
  * By default multi intent model scoring only gives the final prediction which has
  * the probability. This gives the top K intents for analysis
  *
  * @author Neelesh Sambhajiche <neelesh.sa@247-inc.com>
  * @since 22/8/18
  */


/**
  * Params for [[TopKIntents]].
  */
trait TopKIntentsParams extends Params with HasOutputCol  with HasInputCol{

  /**
    * param for the base classifier
    */
  final val stringIndexerModel: Param[StringIndexerModel] = new Param(this, "stringIndexerModel", "String Indexer Model")

  def getStringIndexerModel: StringIndexerModel = $(stringIndexerModel)

  /**
    * param for number of intents
    */
  final val kValue: IntParam = new IntParam(this, "kValue", "Number of Intents")

  def getKValue: Int = $(kValue)
}


object TopKIntentsParams {

  def validateParams(instance: TopKIntentsParams): Unit = {
    def checkElement(elem: Params, name: String): Unit = elem match {
      case _: MLWritable => // good
      case other =>
        throw new UnsupportedOperationException("Uplift write will fail " +
          s" because it contains $name which does not implement MLWritable." +
          s" Non-Writable $name: ${other.uid} of type ${other.getClass}")
    }

    instance match {
      case topKIntents: TopKIntents =>
        checkElement(topKIntents.getStringIndexerModel, "model")
      case _ =>
    }
  }
}

class TopKIntents(override val uid: String)
  extends Transformer
    with TopKIntentsParams
    with MLWritable {

  def this() = this(Identifiable.randomUID("topK"))

  def setStringIndexerModel(value: StringIndexerModel): this.type = {
    set(stringIndexerModel, value.asInstanceOf[StringIndexerModel])
  }

  val labels: StringArrayParam = new StringArrayParam(this,"StringIndexer labels","To identify which label is pointing to which intent name")

  def setLabels(value:Array[String])=set(labels,value)

  def getLabels = ${labels}

  def setKValue(value: Int): this.type = set(kValue, value)

  def setInputCol(value:String="probability"):this.type = set(inputCol,value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def copy(extra: ParamMap): TopKIntents = defaultCopy(extra)

  def transform(df: Dataset[_]): DataFrame = {

    // Get StringIndexerLabels for intents
    val intentLabels = getLabels

    def topIntentUDF(labels: Array[String]) = udf((predictions: Vector) => {
      // get index and prob from array
      val scoresAndLabels =
        for ((score, idx) <- predictions.toArray.zipWithIndex) yield (labels(idx), score)
      scoresAndLabels.sortBy(-_._2).take(getKValue)
    })

    df.withColumn($(outputCol), topIntentUDF(intentLabels)(col("probability")))

  }

  override def transformSchema(schema: StructType): StructType = {

    // Add the return field
    schema.add(StructField($(outputCol), ArrayType(  StructType(Array(StructField("_1", StringType, false),StructField("_2", DoubleType, false))) ), true))

  }

  override def write: MLWriter = new TopKIntents.TopKIntentsWriter(this)

}

object TopKIntents extends MLReadable[TopKIntents] {

  override def read: MLReader[TopKIntents] = new TopKIntentsReader

  override def load(path: String): TopKIntents = super.load(path)

  /** [[MLWriter]] instance for [[TopKIntents]] */
  private[TopKIntents] class TopKIntentsWriter(instance: TopKIntents) extends MLWriter {

    TopKIntentsParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit = {
      val params = instance.extractParamMap().toSeq
      val jsonParams = render(params
        .filter { case ParamPair(p, v) => p.name != "stringIndexerModel" }
        .map { case ParamPair(p, v) => p.name -> parse(p.jsonEncode(v)) }
        .toList)

      DefaultParamsWriter.saveMetadata(instance, path, sc, None, Some(jsonParams))

      val stringIndexerModelPath = new Path(path, s"stringIndexerModel").toString
      instance.getStringIndexerModel.asInstanceOf[MLWritable].save(stringIndexerModelPath)
    }
  }

  private class TopKIntentsReader extends MLReader[TopKIntents] {

    /** Checked against metadata when loading model */
    private val className = classOf[TopKIntents].getName

    override def load(path: String): TopKIntents = {
      implicit val format: DefaultFormats.type = DefaultFormats
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val stringIndexerModelPath = new Path(path, s"stringIndexerModel").toString
      val stringIndexerModel: StringIndexerModel = DefaultParamsReader.loadParamsInstance[StringIndexerModel](stringIndexerModelPath, sc)

      val topKIntents = new TopKIntents(metadata.uid)
      metadata.getAndSetParams(topKIntents)

      topKIntents.setStringIndexerModel(stringIndexerModel)
    }
  }
}