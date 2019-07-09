package com.tfs.test

import java.io.File
import java.io._
import java.nio.file.{FileSystems, Files}
import java.util.concurrent.TimeUnit
import java.util.zip.{ZipEntry, ZipFile, ZipInputStream, ZipOutputStream}

import com.tfs.test.custom.transfomers.CaseNormalizationTransformer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.bundle.SparkBundleContext
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}
import org.scalatest.FlatSpec
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.spark.SparkLeapFrame
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.mleap.TypeConvertersCustom
import resource._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Try

class FlashmlTest extends FlatSpec{

  val config: TSConfig = ConfigFactory.load()

  private val log = LoggerFactory.getLogger(getClass)
  Logger.getLogger("org").setLevel(Level.OFF)

  println("=============================================================================================")
  println("Test case: ML pipeline with file: input3.txt")

  val ss = SparkSession
    .builder()
    .appName("MLPipeline")
    .config("spark.master", "local")
    .getOrCreate()
  import ss.implicits._
  val ds = ss.createDataset[String](Source.fromInputStream(getClass.getResourceAsStream("/input3.txt")).getLines().toSeq)
  val df = ss.read.json(ds).select("text", "id")

  val preprocessing = new Pipeline().setStages(Array(
   // new CaseNormalizationTransformer().setInputCol("text").setOutputCol("text1"),
    new RegexTokenizer().setInputCol("text").setOutputCol("raw").setPattern("\\W").setToLowercase(false),
    new StopWordsRemover().setInputCol("raw").setOutputCol("filtered")
  ))
  val preprocessedModel = preprocessing.fit(df)
  val preprocessedDf = preprocessedModel.transform(df)

  val featuring = new Pipeline().setStages(Array(
    new NGram().setN(2).setInputCol("filtered").setOutputCol("bigrams"),
    new NGram().setN(3).setInputCol("filtered").setOutputCol("trigrams"),
    new NGram().setN(4).setInputCol("filtered").setOutputCol("quadgrams")
  ))
  val featuredModel = featuring.fit(preprocessedDf)
  val featuredDf = featuredModel.transform(preprocessedDf)

  val vectoring = new Pipeline().setStages(Array(
    new HashingTF().setInputCol("filtered").setOutputCol("ughash").setNumFeatures(2000),
    new HashingTF().setInputCol("bigrams").setOutputCol("bghash").setNumFeatures(2000),
    new HashingTF().setInputCol("trigrams").setOutputCol("tghash").setNumFeatures(2000),
    new HashingTF().setInputCol("quadgrams").setOutputCol("qghash").setNumFeatures(2000),
    // Next compute IDF for each of these columns
    new IDF().setInputCol("ughash").setOutputCol("ugidf"),
    new IDF().setInputCol("bghash").setOutputCol("bgidf"),
    new IDF().setInputCol("tghash").setOutputCol("tgidf"),
    new IDF().setInputCol("qghash").setOutputCol("qgidf"),
    //  Next combine these vectors using VectorAssembler
    new VectorAssembler().setInputCols(Array("ugidf", "bgidf", "tgidf", "qgidf")).setOutputCol("features")
  ))
  val vectoredModel = vectoring.fit(featuredDf)
  val vectoredDf = vectoredModel.transform(featuredDf)

  val allStages = ArrayBuffer[PipelineStage]()
  val stringIndex = new StringIndexer().setInputCol("id").setOutputCol("id_label")
  allStages += stringIndex
  val estimator = new LogisticRegression().setMaxIter(10).setLabelCol("id_label").setRegParam(0.1)
  allStages += estimator
  val intermediateModel = new Pipeline().setStages(allStages.toArray).fit(vectoredDf)
  val indexModel = intermediateModel.stages(0).asInstanceOf[StringIndexerModel]
  val indexString = new IndexToString().setInputCol("prediction").setOutputCol("result").setLabels(indexModel.labels)
  allStages += indexString
  val modelling = new Pipeline().setStages(intermediateModel.stages ++ allStages.drop(2))
  /*val modelling = new Pipeline().setStages(Array(
    new StringIndexer().setInputCol("stars").setOutputCol("label"),
    new LogisticRegression().setLabelCol("label").setMaxIter(10).setRegParam(0.1)
  ))*/
  val model = modelling.fit(vectoredDf)

  val newmodel = new Pipeline().setStages(preprocessedModel.stages ++ featuredModel.stages ++ vectoredModel.stages ++ model.stages.drop(1)).fit(df)
  val result = newmodel.transform(df)
  result.show()
  // Save pipeline
  // Make sure that the model folder exist
  val modelDir = new File(s"/home/udhay/tmp/${this.getClass.getSimpleName}").getCanonicalFile
  if(!modelDir.exists()) modelDir.mkdirs()
  // Now delete the specific file
  val file = new File(s"${modelDir.getCanonicalPath}/flashml.zip")
  if(file.exists()) file.delete()

  // Now serialize the pipeline object
  val sbc = SparkBundleContext().withDataset(newmodel.transform(df))
  for(bf <- managed(BundleFile(s"jar:file:${file.getPath}")))
  {   newmodel.writeBundle.save(bf)(sbc).get }
  println(s"Model saved at [${file.getPath}]")

  val bundle = (for(bundleFile <- managed(BundleFile(s"jar:file:${file.getPath}"))) yield
    {   bundleFile.loadMleapBundle().get    }).opt.get
  println(s"Model loaded from [${file.getPath}]")

  val mpipe = bundle.root
  val data = Seq(("The food was awesome, but I didn't like the ambience!!")).toDF("text").toSparkLeapFrame
  mpipe.transform(data).get.toSpark.show()
  val frame = df.toSparkLeapFrame
  val frameres = mpipe.transform(frame).get.toSpark
  frameres.show()

}
