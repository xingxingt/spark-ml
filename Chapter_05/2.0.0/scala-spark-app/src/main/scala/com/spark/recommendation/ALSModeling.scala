package com.spark.recommendation

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.DataFrame

/**
  * Created by manpreet.singh on 07/09/16.
  */
object ALSModeling {

  def createALSModel() {
    //todo 1,获取特征column: userId,movieId,rating
    val ratings = FeatureExtraction.getFeatures()

    //todo 2,拆分数据集 8:2
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    //todo 3,构建ALS模型
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //todo 4,训练ALS模型
    val model = als.fit(training)
    //    model.userFactors.show(false)
    //    model.itemFactors.show(false)
    //todo 5,model.userFactors基于用户推荐模型  model.itemFactors基于物品推荐模型
    println(model.userFactors.count())
    println(model.itemFactors.count())

    //todo 6,使用测试数据集检测模型
    val predictions = model.transform(test)
    println(predictions.printSchema())
    //    predictions.where("userId=13").sort("prediction").show(false)
    import org.apache.spark.sql.functions._
    predictions.where("userId=29").orderBy(desc("prediction")).show(false)

    //todo 7，使用RegressionEvaluator对ALS模型进行评估
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    //
    println(s"Root-mean-square error = $rmse")
  }

  def validateResult(df: DataFrame): Unit = {

    val movies = df.sparkSession.sparkContext.textFile("/Users/axing/Documents/dev/ideaWorkSpace/spark-ml/Chapter_05/2.0.0/scala-spark-app/src/main/scala/com/spark/recommendation/sample_movielens_ratings.txt")

  }

  def main(args: Array[String]) {
    createALSModel()
  }

}
