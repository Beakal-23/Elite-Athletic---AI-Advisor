# Databricks notebook source
#Gold Exercise Recommendations
from pyspark.sql.functions import col, lower, trim, when, lit, concat_ws

gym = spark.table("main.default.silver_gym_exercises")

gold_exercises = (
    gym
    .withColumn("exercise_name", trim(col("exercise_name")))
    .withColumn("body_part", lower(trim(col("body_part"))))
    .withColumn("equipment", lower(trim(col("equipment"))))
    .withColumn("difficulty_level", lower(trim(col("difficulty_level"))))
    .withColumn(
        "goal_tag",
        when(col("body_part").contains("legs"), "speed_agility")
        .when(col("body_part").contains("chest"), "strength")
        .when(col("body_part").contains("back"), "strength")
        .when(col("body_part").contains("shoulders"), "strength")
        .when(col("body_part").contains("abdominals"), "core")
        .otherwise("general_fitness")
    )
    .withColumn(
        "agent_text",
        concat_ws(
            " | ",
            col("exercise_name"),
            col("exercise_type"),
            col("body_part"),
            col("equipment"),
            col("difficulty_level"),
            col("description")
        )
    )
    .filter(col("exercise_name").isNotNull())
)

gold_exercises.write.mode("overwrite").saveAsTable(
    "main.default.gold_exercise_recommendations"
)

display(gold_exercises.limit(10))

# COMMAND ----------

#Gold Basketball Benchmarks

from pyspark.sql.functions import avg, col

nba = spark.table("main.default.silver_nba_stats")

gold_basketball_benchmarks = (
    nba
    .filter(col("season_year") >= 2000)
    .filter(col("position").isNotNull())
    .groupBy("position")
    .agg(
        avg("points").alias("avg_points"),
        avg("assists").alias("avg_assists"),
        avg("rebounds").alias("avg_rebounds"),
        avg("steals").alias("avg_steals"),
        avg("blocks").alias("avg_blocks"),
        avg("minutes_played").alias("avg_minutes_played")
    )
)

gold_basketball_benchmarks.write.mode("overwrite").saveAsTable(
    "main.default.gold_basketball_benchmarks"
)

display(gold_basketball_benchmarks)

# COMMAND ----------

gold_basketball_benchmarks.printSchema()

# COMMAND ----------

#Gold Nutrition Lookup 

food = spark.table("main.default.silver_food_nutrients")

gold_nutrition_lookup = (
    food
    .dropDuplicates()
)

gold_nutrition_lookup.write.mode("overwrite").saveAsTable(
    "main.default.gold_nutrition_lookup"
)

display(gold_nutrition_lookup.limit(10))

# COMMAND ----------

#Gold Saftey Rules -- 

from pyspark.sql import Row

safety_rules = [
    Row(rule_id="R001", rule_category="age", rule_text="Athletes ages 13-17 require parent or coach oversight."),
    Row(rule_id="R002", rule_category="injury", rule_text="If injury status is not none, avoid high-impact training and recommend coach or medical review."),
    Row(rule_id="R003", rule_category="workload", rule_text="Youth athletes should not receive intense training plans for more than 5 days per week."),
    Row(rule_id="R004", rule_category="nutrition", rule_text="Do not recommend supplements, extreme diets, fasting, or medical nutrition treatment."),
    Row(rule_id="R005", rule_category="recovery", rule_text="Every weekly plan must include recovery, stretching, hydration, and sleep guidance."),
    Row(rule_id="R006", rule_category="safety", rule_text="The agent provides educational guidance only, not medical advice.")
]

gold_safety_rules = spark.createDataFrame(safety_rules)

gold_safety_rules.write.mode("overwrite").saveAsTable(
    "main.default.gold_safety_rules"
)

display(gold_safety_rules)

# COMMAND ----------

# Gold food portions

food_portions = spark.table("main.default.silver_food_portions")

gold_food_portions = (
    food_portions
    .dropDuplicates()
)

gold_food_portions.write.mode("overwrite").saveAsTable(
    "main.default.gold_food_portions"
)

display(gold_food_portions.limit(10))

# COMMAND ----------

# Now Let's Verify the Gold Agent Ready Tables -- 

spark.sql("SHOW TABLES IN main.default").show(truncate=False)

# COMMAND ----------

#Quick Agent Readiness Check -- 

gold_tables = [
    "gold_exercise_recommendations",
    "gold_basketball_benchmarks",
    "gold_nutrition_lookup",
    "gold_food_portions",
    "gold_safety_rules"
]

for table in gold_tables:
    df = spark.table(f"main.default.{table}")
    print(f"{table}: {df.count()} rows, {len(df.columns)} columns")