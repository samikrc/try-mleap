package com.tfs.test

import org.scalatest.FlatSpec
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.feature._

import scala.io.Source
import org.apache.spark.sql.SparkSession

class MyTest4 extends FlatSpec
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
    // Start with tokenization and stopword removal
    val pipeline = new Pipeline().setStages(Array(
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
    model.transform(df).select("text", "stars", "result").show(false)

}
