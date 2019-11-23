package org.sparksamples.classification.stumbleupon

import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

/**
  * Created by manpreet.singh on 01/05/16.
  */
object SVMPipeline {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  def svmPipeline(sc: SparkContext) = {
    val records = sc.textFile("/Users/axing/Documents/dev/ideaWorkSpace/spark-ml/Chapter_06/2.0.0/scala-spark-app/src/main/scala/org/sparksamples/classification/dataset/stumbleupon/train_noheader.tsv").map(line => line.split("\t"))

    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }

    // params for SVM
    val numIterations = 10

    // Run training algorithm to build the model
    //训练模型
    val svmModel = SVMWithSGD.train(data, numIterations)

    // Clear the default threshold.
    //去除默认阈值
    svmModel.clearThreshold()

    val svmTotalCorrect = data.map { point =>
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    // calculate accuracy
    //计算精准度
    val svmAccuracy = svmTotalCorrect / data.count()
    println(svmAccuracy)
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local[*]").appName("svm").getOrCreate()
    val sc = spark.sparkContext
    svmPipeline(sc)

  }

}
