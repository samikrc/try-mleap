package com.tfs.test

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
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
import org.apache.spark.sql.mleap.TypeConverters
import resource._

import scala.io.Source
import scala.util.Try

class LogisticRegressionTest extends FlatSpec{

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
  val df = ss.read.json(ds).select("text", "stars")

  // Define the pipeline
  val pipeline = new Pipeline().setStages(Array(
    // Start with tokenization and stopword removal
    new RegexTokenizer().setInputCol("text").setOutputCol("raw").setPattern("\\W").setToLowercase(false),
    new StopWordsRemover().setInputCol("raw").setOutputCol("filtered"),
    // First generate the n-grams: 2-4
    new NGram().setN(2).setInputCol("filtered").setOutputCol("bigrams"),
    new NGram().setN(3).setInputCol("filtered").setOutputCol("trigrams"),
    new NGram().setN(4).setInputCol("filtered").setOutputCol("quadgrams"),
    // Next generate HashingTF from each of the gram columns
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
    new VectorAssembler().setInputCols(Array("ugidf", "bgidf", "tgidf", "qgidf")).setOutputCol("features"),
    // Set up a StringIndexer for the response column
    new StringIndexer().setInputCol("stars").setOutputCol("label"),
    // Now run an One-Vs-Rest SVM model
    new LogisticRegression().setLabelCol("label").setMaxIter(10).setRegParam(0.1),
    // Finally, we need to convert the prediction back to own labels
    new IndexToString().setInputCol("label").setOutputCol("result")
  ))

  val model = pipeline.fit(df)
  // Create a df with only one row to transform and save the pipeline
  val smallDF = df.limit(1).toDF()
  //model.transform(df).show(false)
  val tmp1 = model.transform(df)

  /*val data = ss.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")
  val steps = Array(
    new Tokenizer().setInputCol("text").setOutputCol("words"),
    new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
    new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
  )
  val pipeline = new Pipeline().setStages(steps)
  val model = pipeline.fit(data)
  val predictDf = model.transform(data)*/
  println("Dataset got predicted successfully")

}
