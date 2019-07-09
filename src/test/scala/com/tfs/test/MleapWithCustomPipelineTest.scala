package com.tfs.test

import java.io.{File, FileInputStream}
import java.nio.file.Files
import java.util.zip.GZIPInputStream

import com.tfs.test.Util.Json
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.bundle.SparkBundleContext
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}
import org.scalatest.FlatSpec
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.function.StructSelector
import ml.combust.mleap.spark.SparkLeapFrame
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.mleap.TypeConvertersCustom
import resource._

import scala.io.Source
import scala.util.Try

class MleapWithCustomPipelineTest extends FlatSpec
{
  val config: TSConfig = ConfigFactory.load()

  private val log = LoggerFactory.getLogger(getClass)
  Logger.getLogger("org").setLevel(Level.OFF)

  println("=============================================================================================")
  println("Test case: ML pipeline with file: input3.txt")

  val fs: FileSystem = {
    val hdfsConf = new Configuration()
    hdfsConf.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hdfsConf.set("fs.parameter.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
    hdfsConf.set("fs.defaultFS", "hdfs://localhost:9000")
    FileSystem.get(hdfsConf)
  }

  val ss = SparkSession
    .builder()
    .appName("MLPipeline")
    .config("spark.master", "local")
    .config("hive.metastore.uris","thrift://localhost:9083")
    .enableHiveSupport()
    .getOrCreate()
  import ss.implicits._
  val regexMap = Seq(
    ("_class_hello","\\b(hola|hello|hi|hey)\\b")
  )
  val ds = ss.createDataset[String](Source.fromInputStream(getClass.getResourceAsStream("/input3.txt")).getLines().toSeq)
  val df = ss.read.json(ds).select("text", "stars","cool","useful")
  val wordMap = Map("love"->"_class_feeling","breakfast"-> "_class_food")

  //val df = ss.sql("select * from flashml.NULL_DATA")


  // Define the pipeline
  val pipeline = new Pipeline().setStages(Array(
    // Start with tokenization and stopword removal
    new StopWordsRemoverCustom().setInputCol("text").setOutputCol("text1").setStopWords(Array("the","in","has")).setDelimiter("\\s"),
    new CaseNormalizationTransformer().setInputCol("text1").setOutputCol("text2"),
    new RegexReplacementTransformer().setInputCol("text2").setOutputCol("text3").setRegexReplacements(regexMap),
    new PorterStemmingTransformer().setInputCol("text3").setOutputCol("text4").setExceptions(Array("")).setDelimiter("\\s"),
    new WordSubstitutionTransformer().setInputCol("text4").setOutputCol("text5").setDictionary(wordMap).setDelimiter("\\s"),
    new SentenceMarker().setInputCol("text5").setOutputCol("text6"),
    new RegexTokenizer().setInputCol("text6").setOutputCol("tokens").setPattern("\\s+").setToLowercase(false),
    new NGram().setN(2).setInputCol("tokens").setOutputCol("bigrams"),
    new SkipGramGenerator().setInputCol("tokens").setOutputCol("skipGrams").setWindowSize(4),
    new GramAssembler().setInputCols(Array("tokens","bigrams","skipGrams")).setOutputCol("grams"),
    new HashingTF().setInputCol("tokens").setOutputCol("hash").setNumFeatures(2000),
    new HashingTF().setInputCol("bigrams").setOutputCol("bghash").setNumFeatures(2000),
    new HashingTF().setInputCol("skipGrams").setOutputCol("skiphash").setNumFeatures(2000),
    new IDF().setInputCol("hash").setOutputCol("idf"),
    new IDF().setInputCol("bghash").setOutputCol("bgidf"),
    new IDF().setInputCol("skiphash").setOutputCol("skipidf"),
    new VectorAssembler().setInputCols(Array("idf","bgidf","skipidf")).setOutputCol("features"),
    new StringIndexer().setInputCol("stars").setOutputCol("label"),
    // Now run an One-Vs-Rest SVM model
    new OneVsRestCustom().setClassifier(new LinearSVC().setMaxIter(1).setRegParam(0.1)),
    new PlattScalar().setIsMultiIntent(true).setLabelCol("label")
  ))

  val model = pipeline.fit(df)
  val temp = model.transform(df)
  temp.show()

  // Create a df with only one row to transform and save the pipeline
  val smallDF = df.limit(1).toDF()
  //model.transform(df).show(false)

  // Save pipeline
  // Make sure that the model folder exist
  val modelDir = new File(s"/home/udhay/tmp").getCanonicalFile
  if(!modelDir.exists()) modelDir.mkdirs()
  // Now delete the specific file
  val file = new File(s"/home/udhay/mleap1.zip")
  if(file.exists()) file.delete()

  // Now serialize the pipeline object
  val sbc = SparkBundleContext().withDataset(model.transform(smallDF))
  for(bf <- managed(BundleFile(s"jar:file:${file.getPath}")))
  {   model.writeBundle.save(bf)(sbc).get }
  println(s"Model saved at [${file.getPath}]")

  // Load back the Spark pipeline we saved in the previous section
  val bundle = (for(bundleFile <- managed(BundleFile(s"jar:file:${file.getPath}"))) yield
    {   bundleFile.loadMleapBundle().get    }).opt.get
  println(s"Model loaded from [${file.getPath}]")

  val frame = df.toSparkLeapFrame
  val mpipe = bundle.root
  val temp3 = mpipe.transform(frame).get
  temp3.toSpark.show()

  println("end")

}
