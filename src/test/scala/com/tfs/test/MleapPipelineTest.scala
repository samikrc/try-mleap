package com.tfs.test

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
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
import org.apache.spark.sql.mleap.TypeConverters
import resource._

import scala.io.Source
import scala.util.Try

class MleapPipelineTest extends FlatSpec
{
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
        new OneVsRest().setClassifier(new LinearSVC().setMaxIter(10).setRegParam(0.1)),
        // Finally, we need to convert the prediction back to own labels
        new IndexToString().setInputCol("prediction").setOutputCol("result")
    ))

    val model = pipeline.fit(df)
    // Create a df with only one row to transform and save the pipeline
    val smallDF = df.limit(1).toDF()
    //model.transform(df).show(false)

    // Save pipeline
    // Make sure that the model folder exist
    val modelDir = new File(s"./target/test_models/${this.getClass.getSimpleName}").getCanonicalFile
    if(!modelDir.exists()) modelDir.mkdirs()
    // Now delete the specific file
    val file = new File(s"${modelDir.getCanonicalPath}/spark-pipeline.zip")
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

    import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
    import ml.combust.mleap.core.types._
    /*
    val frame = DefaultLeapFrame(
        smallDF.mleapSchema,
        Seq(Row("The food was awesome, but I didn't like the ambience!!", "4"))
    )
    */
    val frame = df.toSparkLeapFrame
    val mpipe = bundle.root
    mpipe.transform(frame).get.toSpark.show()

    val data = Seq(("The food was awesome, but I didn't like the ambience!!", "1")).toDF("text", "stars").toSparkLeapFrame
    mpipe.transform(data).get.toSpark.show()
}
