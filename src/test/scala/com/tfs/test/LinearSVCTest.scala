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
import org.apache.spark.sql.mleap.TypeConvertersCustom
import resource._

import scala.io.Source
import scala.util.Try

class LinearSVCTest extends FlatSpec
{
    val config: TSConfig = ConfigFactory.load()

    private val log = LoggerFactory.getLogger(getClass)
    Logger.getLogger("org").setLevel(Level.OFF)

    println("=============================================================================================")
    println("Test case: ML pipeline for LinearSVC Model")

    val ss = SparkSession
            .builder()
            .appName("MLPipelineSVC")
            .config("spark.master", "local")
            .getOrCreate()
    import ss.implicits._
    val df = ss.createDataFrame(Seq(
        (0L, "a b c d e spark", 1.0),
        (1L, "b d", 0.0),
        (2L, "spark f g h", 1.0),
        (3L, "hadoop mapreduce", 0.0)
    )).toDF("id", "text", "label")

    // Define the pipeline
    val pipeline = new Pipeline().setStages(Array(
        new Tokenizer().setInputCol("text").setOutputCol("words"),
        new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
        new LinearSVC()
                .setMaxIter(10)
                .setRegParam(0.3)
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
