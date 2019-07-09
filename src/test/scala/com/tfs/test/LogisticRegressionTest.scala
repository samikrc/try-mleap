package com.tfs.test

import java.io.File

import java.io._
import java.nio.file.{FileSystems, Files}
import java.util.concurrent.TimeUnit
import java.util.zip.{ZipEntry, ZipFile, ZipInputStream, ZipOutputStream }

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
import org.apache.spark.sql.mleap.TypeConvertersCustom
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
    new LogisticRegression().setLabelCol("label").setMaxIter(10).setRegParam(0.1)
    // Finally, we need to convert the prediction back to own labels
    //new IndexToString().setInputCol("prediction").setOutputCol("result")
  ))

  val model = pipeline.fit(df)
  val indexMap = model.stages(14).asInstanceOf[StringIndexerModel]
  // Create a df with only one row to transform and save the pipeline
  val smallDF = df.limit(1).toDF()
  //model.transform(df).show(false)
  val tmp1 = model.transform(df)

  val modelDir = new File(s"/home/udhay/tmp/${this.getClass.getSimpleName}").getCanonicalFile
  if(!modelDir.exists()) modelDir.mkdirs()
  // Now delete the specific file
  val file = new File(s"${modelDir.getCanonicalPath}/sp1.zip")
  if(file.exists()) file.delete()

  // Now serialize the pipeline object
  val sbc = SparkBundleContext().withDataset(model.transform(smallDF))
  for(bf <- managed(BundleFile(s"jar:file:${file.getPath}")))
  {   model.writeBundle.save(bf)(sbc).get }
  println(s"Model saved at [${file.getPath}]")

  val pipeline2 = new Pipeline().setStages(Array(new IndexToString().setInputCol("prediction").setOutputCol("result").setLabels(indexMap.labels)))
  val model2 = pipeline2.fit(tmp1)
  val smallDF2 = tmp1.limit(1).toDF()
  val tmp2 = model2.transform(tmp1)

  val modelDir2 = new File(s"/home/udhay/tmp/${this.getClass.getSimpleName}").getCanonicalFile
  if(!modelDir2.exists()) modelDir2.mkdirs()
  // Now delete the specific file
  val file2 = new File(s"${modelDir2.getCanonicalPath}/sp2.zip")
  if(file2.exists()) file2.delete()

  // Now serialize the pipeline object
  val sbc2 = SparkBundleContext().withDataset(model2.transform(smallDF2))
  for(bf <- managed(BundleFile(s"jar:file:${file2.getPath}")))
  {   model2.writeBundle.save(bf)(sbc2).get }
  println(s"Model saved at [${file2.getPath}]")

  def zip(out: String, files: Array[String]) = {
    /*import java.io.{ BufferedInputStream, FileInputStream, FileOutputStream }
    import java.util.zip.{ ZipEntry, ZipOutputStream }*/

    val zip = new ZipOutputStream(new FileOutputStream(out))

    files.foreach { name =>
      zip.putNextEntry(new ZipEntry(name))
      val in = new BufferedInputStream(new FileInputStream(name))
      var b = in.read()
      while (b > -1) {
        zip.write(b)
        b = in.read()
      }
      in.close()
      zip.closeEntry()
    }
    zip.close()
  }

  val file3 = new File(s"${modelDir.getCanonicalPath}/sp3.zip")
  if(file3.exists()) file3.delete()

  zip(file3.getPath,Array(file.getPath,file2.getPath))

  def unZipIt(zipFile: String, outputFolder: String): Unit = {

    val buffer = new Array[Byte](1024)

    try {

      //output directory
      val folder = new File(outputFolder);
      if (!folder.exists()) {
        folder.mkdir();
      }

      //zip file content
      val zis: ZipInputStream = new ZipInputStream(new FileInputStream(zipFile));
      //get the zipped file list entry
      var ze: ZipEntry = zis.getNextEntry();

      while (ze != null) {

        val fileName = ze.getName();
        val newFile = new File(outputFolder + File.separator + fileName);

        System.out.println("file unzip : " + newFile.getAbsoluteFile());

        //create folders
        new File(newFile.getParent()).mkdirs();

        val fos = new FileOutputStream(newFile);

        var len: Int = zis.read(buffer);

        while (len > 0) {

          fos.write(buffer, 0, len)
          len = zis.read(buffer)
        }

        fos.close()
        ze = zis.getNextEntry()
      }

      zis.closeEntry()
      zis.close()

    } catch {
      case e: IOException => println("exception caught: " + e.getMessage)
    }

  }

  val outputDir = new File("/home/udhay/tmp/uncompressed")

  unZipIt(file3.getPath,outputDir.getPath)

  val bundle = (for(bundleFile <- managed(BundleFile(s"jar:file:/home/udhay/tmp/uncompressed/${file.getPath}"))) yield
    {   bundleFile.loadMleapBundle().get    }).opt.get
  println(s"Model loaded from [${file.getPath}]")

  val bundle2 = (for(bundleFile <- managed(BundleFile(s"jar:file:/home/udhay/tmp/uncompressed/${file2.getPath}"))) yield
    {   bundleFile.loadMleapBundle().get    }).opt.get
  println(s"Model loaded from [${file2.getPath}]")

  val mpipe = bundle.root
  val mpipe2 = bundle2.root
  val frame = df.toSparkLeapFrame
  val frameres = mpipe.transform(frame).get
  val frame2res = mpipe2.transform(frameres).get.toSpark
  frame2res.show()

  println("Dataset got predicted successfully")

}
