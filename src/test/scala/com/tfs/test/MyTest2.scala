package com.tfs.test

import org.scalatest.FlatSpec
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}

import scala.io.Source
import org.apache.spark.sql.SparkSession
import com.tfs.test.SparkWordCounter

class MyTest2 extends FlatSpec
{

    val config: TSConfig = ConfigFactory.load()

    private val log = LoggerFactory.getLogger(getClass)
    Logger.getLogger("org").setLevel(Level.OFF)

    println("=============================================================================================")
    println("Test case: Word count with file: input2.txt")

    val ss = SparkSession
            .builder()
            .appName("WordCount2")
            .config("spark.master", "local")
            .getOrCreate()
    val fileStream = MyTest2.this.getClass.getResourceAsStream("/input2.txt")
    val counts = SparkWordCounter.countWords(ss.sparkContext, Source.fromInputStream(fileStream).getLines().toList)

    "Count of words" should "match" in
    {
        assert(counts("name") == 4)
        assert(counts("Suites") == 5)
    }
}
