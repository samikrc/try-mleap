package com.tfs.test

import org.apache.spark.ml.classification._
import org.apache.spark.ml.clustering._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression._

/**
  * Class to test the mleap pileline for a series of testcases.
  * Directly inspired by the test suite available in the project spark-ml-serving,
  * available at: https://github.com/Hydrospheredata/spark-ml-serving
  * Look at: https://github.com/Hydrospheredata/spark-ml-serving/tree/master/src/test/scala/io/hydrosphere/spark_ml_serving
  */
class CommonModelsSpec extends GenericTestSpec
{
    modelTest(
        data = session.createDataFrame(
            Seq(
                (0.0, "Hi I heard about Spark"),
                (0.0, "I wish Java could use case classes"),
                (1.0, "Logistic regression models are neat")
            )
        ).toDF("label", "sentence"),
        steps = Seq(
            new Tokenizer().setInputCol("sentence").setOutputCol("words"),
            new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20),
            new IDF().setInputCol("rawFeatures").setOutputCol("features")
        ),
        columns = Seq(
            "features"
        )
    )

    modelTest(
        data = session.createDataFrame(
            Seq(
                (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
                (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
                (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
            )
        ).toDF("id", "features", "clicked"),
        steps = Seq(
            new ChiSqSelector().setNumTopFeatures(1).setFeaturesCol("features").setLabelCol("clicked").setOutputCol("selectedFeatures")

        ),
        columns = Seq(
            "selectedFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Array("a", "b", "c")),
            (1, Array("a", "b", "b", "c", "a"))
        )).toDF("id", "words"),
        steps = Seq(
            new CountVectorizer()
                    .setInputCol("words")
                    .setOutputCol("features")
                    .setVocabSize(3)
                    .setMinDF(2)
        ),
        columns = Seq(
            "features"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Array("Provectus", "is", "such", "a", "cool", "company")),
            (1, Array("Big", "data", "rules", "the", "world")),
            (2, Array("Cloud", "solutions", "are", "our", "future"))
        )).toDF("id", "words"),
        steps = Seq(
            new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
        ),
        columns = Seq(
            "ngrams"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            Vectors.dense(0.0, 10.3, 1.0, 4.0, 5.0),
            Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
            Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
        ).map(Tuple1.apply)).toDF("features"),
        steps = Seq(
            new StandardScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledFeatures")
                    .setWithStd(true)
                    .setWithMean(false)
        ),
        columns = Seq(
            "scaledFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Seq("I", "saw", "the", "red", "balloon")),
            (1, Seq("Mary", "had", "a", "little", "lamb"))
        )).toDF("id", "raw"),
        steps = Seq(
            new StopWordsRemover()
                    .setInputCol("raw")
                    .setOutputCol("filtered")
        ),
        columns = Seq(
            "filtered"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Vectors.dense(1.0, 0.1, -8.0)),
            (1, Vectors.dense(2.0, 1.0, -4.0)),
            (2, Vectors.dense(4.0, 10.0, 8.0))
        )).toDF("id", "features"),
        steps = Seq(
            new MaxAbsScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledFeatures")
        ),
        columns = Seq(
            "scaledFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Vectors.dense(1.0, 0.1, -1.0)),
            (1, Vectors.dense(2.0, 1.1, 1.0)),
            (2, Vectors.dense(3.0, 10.1, 3.0))
        )).toDF("id", "features"),
        steps = Seq(
            new MinMaxScaler()
                    .setInputCol("features")
                    .setOutputCol("scaledFeatures")
        ),
        columns = Seq(
            "scaledFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, "a"), (1, "b"), (2, "c"),
            (3, "a"), (4, "a"), (5, "c")
        )).toDF("id", "category"),
        steps = Seq(
            new StringIndexer()
                    .setInputCol("category")
                    .setOutputCol("categoryIndex"),
            new OneHotEncoderEstimator()
                    .setInputCols(Array("categoryIndex"))
                    .setOutputCols(Array("categoryVec"))
        ),
        columns = Seq(
            "categoryVec"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
            Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
            Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
        ).map(Tuple1.apply)).toDF("features"),
        steps = Seq(
            new PCA()
                    .setInputCol("features")
                    .setOutputCol("pcaFeatures")
                    .setK(3)
        ),
        columns = Seq(
            "pcaFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, Vectors.dense(1.0, 0.5, -1.0)),
            (1, Vectors.dense(2.0, 1.0, 1.0)),
            (2, Vectors.dense(4.0, 10.0, 2.0))
        )).toDF("id", "features"),
        steps = Seq(
            new Normalizer()
                    .setInputCol("features")
                    .setOutputCol("normFeatures")
                    .setP(1.0)
        ),
        columns = Seq(
            "normFeatures"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            Vectors.dense(0.0, 1.0, -2.0, 3.0),
            Vectors.dense(-1.0, 2.0, 4.0, -7.0),
            Vectors.dense(14.0, -2.0, -5.0, 1.0)
        ).map(Tuple1.apply)).toDF("features"),
        steps = Seq(
            new DCT()
                    .setInputCol("features")
                    .setOutputCol("featuresDCT")
                    .setInverse(false)
        ),
        columns = Seq(
            "featuresDCT"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new NaiveBayes()
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, 0.1),
            (1, 0.8),
            (2, 0.2)
        )).toDF("id", "feature"),
        steps = Seq(
            new Binarizer()
                    .setInputCol("feature")
                    .setOutputCol("binarized_feature")
                    .setThreshold(5.0)
        ),
        columns = Seq(
            "binarized_feature"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps =
                Seq(
                    new StringIndexer()
                            .setInputCol("label")
                            .setOutputCol("indexedLabel"),
                    new VectorIndexer()
                            .setInputCol("features")
                            .setOutputCol("indexedFeatures")
                            .setMaxCategories(4),
                    new GBTClassifier()
                            .setLabelCol("indexedLabel")
                            .setFeaturesCol("indexedFeatures")
                            .setMaxIter(10)
                ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new StringIndexer()
                    .setInputCol("label")
                    .setOutputCol("indexedLabel"),
            new VectorIndexer()
                    .setInputCol("features")
                    .setOutputCol("indexedFeatures")
                    .setMaxCategories(4),
            new DecisionTreeClassifier()
                    .setLabelCol("indexedLabel")
                    .setFeaturesCol("indexedFeatures")
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0L, "a b c d e spark", 1.0),
            (1L, "b d", 0.0),
            (2L, "spark f g h", 1.0),
            (3L, "hadoop mapreduce", 0.0)
        )).toDF("id", "text", "label"),
        steps = Seq(
            new Tokenizer().setInputCol("text").setOutputCol("words"),
            new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
            new LinearRegression()
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0L, "a b c d e spark", 1.0),
            (1L, "b d", 0.0),
            (2L, "spark f g h", 1.0),
            (3L, "hadoop mapreduce", 0.0)
        )).toDF("id", "text", "label"),
        steps = Seq(
            new Tokenizer().setInputCol("text").setOutputCol("words"),
            new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
            new LinearSVC()
                    .setMaxIter(10)
                    .setRegParam(0.3)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
            new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new StringIndexer().setInputCol("label").setOutputCol("indexedLabel"),
            new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
            new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4),
            new RandomForestRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
        ),
        columns = Seq(
            "prediction"
        )
    )


    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new VectorIndexer()
                    .setInputCol("features")
                    .setOutputCol("indexedFeatures")
                    .setMaxCategories(4),
            new GBTRegressor()
                    .setLabelCol("label")
                    .setFeaturesCol("indexedFeatures")
                    .setMaxIter(10)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new KMeans().setK(2).setSeed(1L)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (Vectors.dense(4.0, 0.2, 3.0, 4.0, 5.0), 1.0),
            (Vectors.dense(3.0, 0.3, 1.0, 4.1, 5.0), 1.0),
            (Vectors.dense(2.0, 0.5, 3.2, 4.0, 5.0), 1.0),
            (Vectors.dense(5.0, 0.7, 1.5, 4.0, 5.0), 1.0),
            (Vectors.dense(1.0, 0.1, 7.0, 4.0, 5.0), 0.0),
            (Vectors.dense(8.0, 0.3, 5.0, 1.0, 7.0), 0.0)
        )).toDF("features", "label"),
        steps = Seq(
            new GaussianMixture().setK(2)
        ),
        columns = Seq(
            "prediction"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq(
            (0, "Hi I heard about Spark"),
            (1, "I wish Java could use case classes"),
            (2, "Logistic,regression,models,are,neat")
        )).toDF("id", "sentence"),
        steps = Seq(
            new RegexTokenizer()
                    .setInputCol("sentence")
                    .setOutputCol("words")
                    .setPattern("\\W")
        ),
        columns = Seq(
            "words"
        )
    )

    modelTest(
        data = session.createDataFrame(Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
        ).toDF("id", "hour", "mobile", "userFeatures", "clicked"),
        steps = Seq(
            new VectorAssembler()
                    .setInputCols(Array("hour", "mobile", "userFeatures"))
                    .setOutputCol("features")
        ),
        columns = Seq(
            "features"
        )
    )

    modelTest(
        data = session.read.format("libsvm")
                .load(getClass.getResource("/sample_lda_libsvm_data.txt").getPath),
        steps = Seq(
            new LDA().setK(10).setMaxIter(10)
        ),
        columns = Seq(
            "topicDistribution"
        ),
        accuracy = 1
    )

    modelTest(
        data = session.read.json(getClass.getResource("/input3.txt").getPath).select("text"),
        steps = Seq(
            new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\W").setToLowercase(false),
            new Word2Vec().setInputCol("words").setOutputCol("result").setVectorSize(10).setMinCount(0)
        ),
        columns = Seq("result"),
        accuracy = 1
    )

    modelTest(
        "LargeMLPipeline",
        data = session.read.json(getClass.getResource("/input3.txt").getPath).select("text", "stars"),
        steps = Seq(
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
        ),
        columns = Seq("result"),
        accuracy = 1
    )
}
