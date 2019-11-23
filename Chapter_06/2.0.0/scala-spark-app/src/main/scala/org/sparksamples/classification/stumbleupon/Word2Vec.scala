package org.sparksamples.classification.stumbleupon

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

/**
  * Word2Vec : 将单词表示为向量，可以计算两个单词的相似度
  * 使用文档中每个词语的平均数来将文档转换为向量，然后这个向量可以作为预测的特征，来计算文档相似度计算
  */
object Word2Vec {

  def main(args: Array[String]): Unit = {


    // a. 构建SparkSession实例对象
    val spark = SparkSession.builder()
      .appName("NewsCategoryPredictNBTest")
      .master("local[4]")
      .getOrCreate()


    // Input data: Each row is a bag of words from a sentence or document.
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    documentDF.show(false)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.select("result").take(3).foreach(println)

  }

}
