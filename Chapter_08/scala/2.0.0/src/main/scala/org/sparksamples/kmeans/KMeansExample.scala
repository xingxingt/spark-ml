/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.sparksamples.kmeans

// scalastyle:off println

// $example on$
import org.apache.spark.SparkConf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
// $example off$
import org.apache.spark.sql.SparkSession

/**
  * An example demonstrating k-means clustering.
  * Run with
  * {{{
  * bin/run-example ml.KMeansExample
  * }}}
  */
object KMeansExample {
  val PATH = "/Users/axing/Documents/dev/ideaWorkSpace/spark-ml/Chapter_08/scala/2.0.0/"


  def isColumnNameLine(line: String): Boolean = {
    if (line != null && line.contains("Channel")) true
    else false
  }

  def main(args: Array[String]): Unit = {
    // Creates a SparkSession.
    //val spark = SparkSession
    //  .builder
    //  .appName(s"${this.getClass.getSimpleName}")
    //  .getOrCreate()

    val spConfig = (new SparkConf).setMaster("local[1]").setAppName("SparkApp").
      set("spark.driver.allowMultipleContexts", "true")

    val spark = SparkSession
      .builder()
      .appName("Spark SQL Example")
      .config(spConfig)
      .getOrCreate()

    // $example on$
    // Loads data.
    val inputDf = spark.read.option("header", "true").csv(PATH + "data/Wholesale customers data.csv")
    import spark.implicits._
    val dataset: Dataset[LabeledPoint] = inputDf.map(row => {
      val array = Array(row.getString(1).toDouble, row.getString(2).toDouble, row.getString(3).toDouble,
        row.getString(4).toDouble,
        row.getString(5).toDouble, row.getString(6).toDouble, row.getString(7).toDouble)
      // 特征
      val features = Vectors.dense(array)
      // 返回标签向量
      LabeledPoint(row.getString(0).toDouble, features)
    })
    dataset.show()

    val splitDataSet = dataset.randomSplit(Array(0.8, 0.2))
    val trainData = splitDataSet(0)
    val testData = splitDataSet(1)
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(trainData)

    // Make predictions
    val predictions = model.transform(testData)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()
    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    //ps:同样的迭代次数和算法跑的次数，这个值越小代表聚类的效果越好,用来选择K值
    //    val WSSSE = model.computeCost(dataset)
    //    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
    // $example off$

    spark.stop()
  }
}

// scalastyle:on println
