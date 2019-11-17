package com.spark.recommendation

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.DataFrame

/**
  * Created by xinghe on 07/09/16.
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
      .setRank(5) //推荐前5
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //todo 4,训练ALS模型
    val model = als.fit(training)
    //用户因子矩阵
    //    model.userFactors.show(false)
    //物品因子矩阵
    //    model.itemFactors.show(false)
    //todo 5,model.userFactors基于用户推荐模型  model.itemFactors基于物品推荐模型
    //    println(model.userFactors.count())
    //    println(model.itemFactors.count())

    //todo 6,使用测试数据集检测模型
    val predictions = model.transform(test)
    println(predictions.printSchema())


    //todo 使用ALS模型进行推荐
    println(s"=============================基于用户推荐模型=============================")
    model.recommendForAllUsers(5).show(false)

    println(s"==============================基于物品推荐模型============================")
    model.recommendForAllItems(5).show(false)


    //todo 7，使用RegressionEvaluator对ALS模型进行评估
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    //
    println(s"Root-mean-square error = $rmse")


    //todo 模型保存/读取模型
    //    model.save("/Users/axing/Documents/dev/ideaWorkSpace/spark-ml/Chapter_05/2.0.0/scala-spark-app/src/data/model/alsModel")
  }

  def validateResult(df: DataFrame): Unit = {

    val movies = df.sparkSession.sparkContext.textFile("/Users/axing/Documents/dev/ideaWorkSpace/spark-ml/Chapter_05/2.0.0/scala-spark-app/src/main/scala/com/spark/recommendation/sample_movielens_ratings.txt")

  }

  def main(args: Array[String]) {
    createALSModel()
  }

}
