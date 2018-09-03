package com.tfs.test

import org.scalatest.FlatSpec
import org.apache.log4j.{Level, Logger}
import org.slf4j.LoggerFactory
import com.typesafe.config.{ConfigFactory, Config => TSConfig}

import scala.io.Source
import org.apache.spark.sql.SparkSession

class MyTest3 extends FlatSpec
{
    val config: TSConfig = ConfigFactory.load()

    private val log = LoggerFactory.getLogger(getClass)
    Logger.getLogger("org").setLevel(Level.OFF)

    println("=============================================================================================")
    println("Test case: Word count with file: input3.txt")

    val ss = SparkSession
            .builder()
            .appName("WordCount3")
            .config("spark.master", "local")
            .getOrCreate()
    val df = ss.read.json(getClass.getResource("/input3.txt").getPath).select("text", "stars")
    df.printSchema()

    "Count of columns" should "match" in {  assert(df.columns.length == 2)   }
    "Count of records" should "match" in {  assert(df.count() == 20)    }
}
