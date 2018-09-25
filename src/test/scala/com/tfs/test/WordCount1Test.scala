package com.tfs.test

import org.scalatest.FlatSpec
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}

import scala.io.Source
import org.apache.spark.sql.SparkSession

class WordCount1Test extends FlatSpec
{

    val config: TSConfig = ConfigFactory.load()

    private val log = LoggerFactory.getLogger(getClass)
    Logger.getLogger("org").setLevel(Level.OFF)

    println("=============================================================================================")
    println("Test case: Word count with file: input1.txt")

    val ss = SparkSession
            .builder()
            .appName("WordCount1")
            .config("spark.master", "local")
            .getOrCreate()
    val fileStream = getClass.getResourceAsStream("/input1.txt")
    val input = ss.sparkContext.makeRDD(Source.fromInputStream(fileStream).getLines().toList)
    val counts = input.flatMap(line ⇒ line.split(" "))
            .map(word ⇒ (word, 1))
            .reduceByKey(_ + _)
            .collect()
            .toMap

    "Count of words" should "match" in
    {
        assert(counts("package") == 3)
        assert(counts("classes") == 4)
        assert(counts("loaded") == 3)
    }
}
