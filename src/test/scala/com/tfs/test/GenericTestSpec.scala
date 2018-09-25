package com.tfs.test

import java.io.File
import java.nio.file.Path

import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.linalg.{Matrix => OldMatrix, Vector => OldVector}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSpec}
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.Transformer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.mleap.TypeConverters
import resource._

trait GenericTestSpec extends FunSpec with BeforeAndAfterAll
{
    Logger.getLogger("org").setLevel(Level.OFF)
    val conf = new SparkConf()
            .setMaster("local[2]")
            .setAppName("test")
            .set("spark.ui.enabled", "false")
    val session: SparkSession = SparkSession.builder().config(conf).getOrCreate()

    lazy val modelPath = s"./target/test_models/${session.version}"
    //def modelPath(modelName: String): String = s"./target/test_models/${session.version}/$modelName"
    //def modelPath(modelName: String): String = new File(s"/tmp/$modelName").getCanonicalPath

    def test(
        name: String,
        data: => DataFrame,
        steps: => Seq[PipelineStage],
        columns: => Seq[String],
        accuracy: Double = 0.01
    ) =
    {
        var validation: DataFrame = null
        var localPipelineModel: Transformer = null

        // Make sure that the model folder exist
        val dir = new File(modelPath).getCanonicalFile
        if(!dir.exists()) dir.mkdirs()
        // Now delete the specific file
        val file = new File(s"${dir.getCanonicalPath}/${name.toLowerCase}.zip")
        if(file.exists()) file.delete()

        it("should train")
        {
            val pipeline = new Pipeline().setStages(steps.toArray)
            val pipelineModel = pipeline.fit(data)
            validation = pipelineModel.transform(data)
            // Save the pipeline
            val sbc = SparkBundleContext().withDataset(validation)
            for(bf <- managed(BundleFile(s"jar:file:${file.getPath}")))
            {   pipelineModel.writeBundle.save(bf)(sbc).get }
            println(s"Model saved at [${file.getPath}]")
        }

        it("should load local version")
        {
            println(s"Loading model from: [${file.getPath}]")
            localPipelineModel = (for(bundleFile <- managed(BundleFile(s"jar:file:${file.getPath}"))) yield
            {   bundleFile.loadMleapBundle().get    }).opt.get.root
            println("Model loaded")
            assert(localPipelineModel != null)
        }

        it("should transform LocalData")
        {
            val localData = data.toSparkLeapFrame
            val result = localPipelineModel.transform(localData).get.toSpark

            columns.foreach
            { col =>
                val resCol = result.select(col).collect()
                        //.getOrElse(throw new IllegalArgumentException("Result column is absent"))
                val valCol = validation.select(col).collect()
                        //.getOrElse(throw new IllegalArgumentException("Validation column is absent"))
                resCol.zip(valCol).foreach
                {
                    case (r: Seq[Number@unchecked], v: Seq[Number@unchecked]) if r.head.isInstanceOf[Number] && v.head.isInstanceOf[Number] =>
                        r.zip(v).foreach
                        {
                            case (ri, vi) =>
                                assert(ri.doubleValue() - vi.doubleValue() <= accuracy, s"$ri - $vi > $accuracy")
                        }
                    case (r: Number, v: Number) =>
                        assert(r.doubleValue() - v.doubleValue() <= accuracy, s"$r - $v > $accuracy")
                    case (r, n) =>
                        assert(r === n)
                }
                /*
                // Skip type related tests
                result.select(col).foreach
                { resData =>
                    resData.foreach
                    { resRow =>
                        if (resRow.isInstanceOf[Seq[_]])
                        {
                            assert(resRow.isInstanceOf[List[_]], resRow)
                        }
                        else if (resRow.isInstanceOf[Vector] || resRow.isInstanceOf[OldVector] || resRow
                                .isInstanceOf[Matrix] || resRow.isInstanceOf[OldMatrix])
                        {
                            assert(false, s"SparkML type detected. Column: $col, value: $resRow")
                        }
                    }
                }
                */
            }
        }
    }

    def modelTest(
         name: String = "",
         data: => DataFrame,
         steps: => Seq[PipelineStage],
         columns: => Seq[String],
         accuracy: Double = 0.01
     ): Unit =
    {
        lazy val testName = if(name.isEmpty) steps.map(_.getClass.getSimpleName).foldLeft("")
        {
            case ("", b) => b
            case (a, b) => a + "-" + b
        } else name

        describe(testName)
        {
            test(testName, data, steps, columns, accuracy)
        }
    }
}
