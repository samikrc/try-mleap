package com.tfs.test

import org.apache.spark.SparkContext

object SparkWordCounter
{
    def countWords(sc: SparkContext, wordList: List[String]) =
    {
        val input = sc.parallelize(wordList)
        input.flatMap(line ⇒ line.split(" "))
            .map(word ⇒ (word, 1))
            .reduceByKey(_ + _)
            .collect()
            .toMap
    }
}
