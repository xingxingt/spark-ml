package org.sparksamples.classification.stumbleupon

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TFIdf {


  def main(args: Array[String]): Unit = {
    // a. 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("NewsCategoryPredictNBTest")
      .master("local[4]")
      .getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    //构建分词器
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.printSchema()
    //    root
    //    |-- label: integer (nullable = false)
    //    |-- sentence: string (nullable = true)
    //    |-- words: array (nullable = true)
    //    |    |-- element: string (containsNull = true)

    wordsData.show(false)


    //将句子转换为特征向量使用hashingTF
    val hashingTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.show(false)
    featurizedData.printSchema()
    //    root
    //    |-- label: integer (nullable = false)
    //    |-- sentence: string (nullable = true)
    //    |-- words: array (nullable = true)
    //    |    |-- element: string (containsNull = true)
    //    |-- rawFeatures: vector (nullable = true)

    // alternatively, CountVectorizer can also be used to get term frequency vectors
    // tf to idf
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)



  }

}
