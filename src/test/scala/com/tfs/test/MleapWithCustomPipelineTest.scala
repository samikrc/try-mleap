package com.tfs.test

import java.io.{File, FileInputStream}
import java.nio.file.Files
import java.util.zip.GZIPInputStream

import com.tfs.flashml.core.featuregeneration.transformer.CategoricalColumnsTransformer
import com.tfs.test.Util.Json
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Model, Pipeline, feature}
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
import org.apache.spark.sql.functions.col
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
  val ds = ss.createDataset[String](Source.fromInputStream(getClass.getResourceAsStream("/input4.txt")).getLines().toSeq)
  val df = ss.read.json(ds).select("text", "stars","cool","useful")
  //val df = ss.sql("select * from flashml.hilton_ci_train limit 20")
  /*val df = ss.sqlContext.read.format("csv")
  .option("header",true)
  .load(getClass.getResource("/spark_demo.csv").getPath)
  .withColumn("test_double",col("test_double").cast("double"))*/
  val wordMap = Map("love"->"_class_feeling","breakfast"-> "_class_food")

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
    new VectorAssembler().setInputCols(Array("idf","bgidf","skipidf","useful")).setOutputCol("features"),
    new StringIndexer().setInputCol("stars").setOutputCol("label"),
    // Now run an One-Vs-Rest SVM model
    //new LogisticRegression().setMaxIter(1).setRegParam(0.1)
    new LinearSVC().setMaxIter(1).setRegParam(0.1),
    //new OneVsRestCustom().setClassifier(new LinearSVC().setMaxIter(1).setRegParam(0.1)),
    new PlattScalar().setIsMultiIntent(false).setLabelCol("label")
   /* new ImputerCustom().setInputCol("test_string").setReplacementValue("other"),
    new ImputerCustom().setInputCol("test_double").setReplacementValue("0")*/

  ))

  val model = pipeline.fit(df)
  val temp = model.transform(df)
  temp.show()
  val pipeline2 = new Pipeline().setStages(Array(new UpliftTransformer().setBaseClassifier(model.stages(18).asInstanceOf[Model[_]]).setPlattScalar(model.stages(19).asInstanceOf[PlattScalarModel])))
  val model2 = pipeline2.fit(temp)
  val temp2 = model2.transform(temp)

  val finalPipeline = new Pipeline().setStages(model.stages ++ model2.stages)
  val finalModel = finalPipeline.fit(df)
  val finalDf = finalModel.transform(df)
  finalDf.show()
  //model.transform(df).show(false)

  // Create a df with only one row to transform and save the pipeline
  val smallDF = df.limit(1).toDF()

  // Save pipeline
  // Make sure that the model folder exist
  val modelDir = new File(s"/home/udhay/tmp").getCanonicalFile
  if(!modelDir.exists()) modelDir.mkdirs()
  // Now delete the specific file
  val file = new File(s"/home/udhay/mleap1.zip")
  if(file.exists()) file.delete()

  // Now serialize the pipeline object
  val sbc = SparkBundleContext().withDataset(finalModel.transform(smallDF))
  for(bf <- managed(BundleFile(s"jar:file:${file.getPath}")))
  {   finalModel.writeBundle.save(bf)(sbc).get }
  println(s"Model saved at [${file.getPath}]")

  //val file = new File(s"/home/udhay/mleap1.zip")
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
