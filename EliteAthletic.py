# Databricks notebook source
import pandas as pd
import random

positions = [
    "Point Guard",
    "Shooting Guard",
    "Small Forward",
    "Power Forward",
    "Center"
]

experience_levels = [
    "Beginner",
    "Intermediate",
    "Advanced"
]

goals = [
    "Improve shooting",
    "Increase speed",
    "Improve ball handling",
    "Increase vertical jump",
    "Improve conditioning",
    "Improve defense",
    "Build strength",
    "Improve agility"
]

injuries = [
    "None",
    "Ankle Recovery",
    "Knee Recovery",
    "Minor Shoulder Pain"
]

equipment_options = [
    "Basketball",
    "Basketball, Cones",
    "Basketball, Resistance Bands",
    "Basketball, Gym Access",
    "Basketball, Cones, Gym Access",
    "Basketball, Cones, Resistance Bands, Gym Access"
]

profiles = []

for i in range(1, 31):

    athlete = {
        "athlete_id": f"A{i:03}",
        "age": random.randint(13, 17),
        "sport": "Basketball",
        "position": random.choice(positions),
        "experience_level": random.choice(experience_levels),
        "goal": random.choice(goals),
        "injury_status": random.choice(injuries),
        "available_days": random.randint(2, 6),
        "equipment": random.choice(equipment_options),
        "notes": "Parent/Coach Oversight Required"
    }

    profiles.append(athlete)

df = pd.DataFrame(profiles)

display(df)

# COMMAND ----------

csv_path = "/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project/athlete_test_profiles.csv"

df.to_csv(csv_path, index=False)

print(f"Saved to: {csv_path}")

# COMMAND ----------

import os

os.path.exists("/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project/athlete_test_profiles.csv")

# COMMAND ----------

# --Load Back Into Spark -- 
athlete_df = spark.read.csv(
    "/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project/athlete_test_profiles.csv",
    header=True,
    inferSchema=True
)

display(athlete_df)

# COMMAND ----------

# Save as a Bronze Table -- This is what we'll use later for your agent

athlete_df.write.mode("overwrite").saveAsTable(
    "main.default.bronze_athlete_profiles"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.bronze_athlete_profiles
# MAGIC LIMIT 10;

# COMMAND ----------

#Setting the Catalog/Schema 

spark.sql("USE CATALOG main")
spark.sql("USE SCHEMA default")

# COMMAND ----------

#Defining the file path 
base_path = "dbfs:/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project"

files = {
    "bronze_athlete_profiles": "athlete_test_profiles.csv",
    "bronze_player_data": "player_data.csv",
    "bronze_players": "Players.csv",
    "bronze_nba_stats": "Seasons_Stats.csv",
    "bronze_gym_exercises": "megaGymDataset.csv",
    "bronze_food_sample": "sub_sample_food.csv",
    "bronze_food_nutrients": "sub_sample_result.csv",
    "bronze_food_categories": "food_category.csv",
    "bronze_food_portions": "food_portion.csv"
}

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project/"))

# COMMAND ----------

base_path = (
    "dbfs:/Workspace/Users/bzekaryas@sandiego.edu/"
    "Elite Athletic - Final Project/"
)

# COMMAND ----------

#Quick test -- 
nba_df = spark.read.csv(
    base_path + "Seasons_Stats.csv",
    header=True,
    inferSchema=True
)

display(nba_df)

# COMMAND ----------

Athlete profiles complete -- Bronze data layer inprogress -- Bronze athlet profiles complete

# COMMAND ----------

#Moving to Data Profile & Data Cleaning -- Before building the silver layer
#Generic Data Profiler -- 

from pyspark.sql.functions import col, count, when

def profile_table(table_name):

    print("=" * 80)
    print(f"TABLE: {table_name}")
    print("=" * 80)

    df = spark.table(table_name)

    print(f"\nRows: {df.count()}")
    print(f"Columns: {len(df.columns)}")

    print("\nSchema:")
    df.printSchema()

    print("\nNull Counts:")

    null_counts = df.select([
        count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])

    display(null_counts)

    print("\nDuplicate Count:")

    duplicates = (
        df.count()
        - df.dropDuplicates().count()
    )

    print(f"Duplicates: {duplicates}")

    print("\nSample Records:")

    display(df.limit(5))

# COMMAND ----------

profile_table(
    "main.default.bronze_athlete_profiles"
)

# COMMAND ----------

spark.sql("SHOW TABLES IN main.default").show(truncate=False)

# COMMAND ----------

athlete_df = spark.table("main.default.bronze_athlete_profiles")
display(athlete_df)

# COMMAND ----------

base_path = "dbfs:/Workspace/Users/bzekaryas@sandiego.edu/Elite Athletic - Final Project/"

files = {
    "bronze_nba_stats": "Seasons_Stats.csv",
    "bronze_player_data": "player_data.csv",
    "bronze_players": "Players.csv",
    "bronze_gym_exercises": "megaGymDataset.csv",
    "bronze_food_sample": "sub_sample_food.csv",
    "bronze_food_nutrients": "sub_sample_result.csv",
    "bronze_food_categories": "food_category.csv",
    "bronze_food_portions": "food_portion.csv"
}

for table_name, file_name in files.items():
    path = base_path + file_name

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )

    df.write.mode("overwrite").saveAsTable(f"main.default.{table_name}")

    print(f"Created: main.default.{table_name}")

# COMMAND ----------

#This should show the Bronze layer is now complete 
spark.sql("SHOW TABLES IN main.default").show(truncate=False)

# COMMAND ----------


# Running the profiler -- 
tables_to_profile = [
    "main.default.bronze_athlete_profiles",
    "main.default.bronze_nba_stats",
    "main.default.bronze_gym_exercises",
    "main.default.bronze_food_nutrients",
    "main.default.bronze_food_portions"
]

for table in tables_to_profile:
    profile_table(table)

# COMMAND ----------

#Raw count summary -- 
for table in tables_to_profile:
    df = spark.table(table)
    print(f"{table}: {df.count()} rows, {len(df.columns)} columns")


# COMMAND ----------

#setting Catalog/Schema -- 
spark.sql("USE CATALOG main")
spark.sql("USE SCHEMA default")


# COMMAND ----------

#Silvering Athlete Profiles
from pyspark.sql.functions import col, trim, lower

athlete = spark.table("main.default.bronze_athlete_profiles")

silver_athlete = (
    athlete
    .dropDuplicates(["athlete_id"])
    .withColumn("sport", lower(trim(col("sport"))))
    .withColumn("position", trim(col("position")))
    .withColumn("experience_level", lower(trim(col("experience_level"))))
    .withColumn("injury_status", lower(trim(col("injury_status"))))
    .withColumn("available_days", col("available_days").cast("int"))
    .filter((col("age") >= 13) & (col("age") <= 17))
)

silver_athlete.write.mode("overwrite").saveAsTable(
    "main.default.silver_athlete_profiles"
)

display(silver_athlete)

# COMMAND ----------

#silver NBA Stats -- 

nba = spark.table("main.default.bronze_nba_stats")

silver_nba = (
    nba
    .dropDuplicates()
    .select(
        col("Year").cast("int").alias("season_year"),
        col("Player").alias("player_name"),
        col("Pos").alias("position"),
        col("Age").cast("int").alias("age"),
        col("G").cast("int").alias("games"),
        col("MP").cast("double").alias("minutes_played"),
        col("PTS").cast("double").alias("points"),
        col("AST").cast("double").alias("assists"),
        col("TRB").cast("double").alias("rebounds"),
        col("STL").cast("double").alias("steals"),
        col("BLK").cast("double").alias("blocks")
    )
    .filter(col("player_name").isNotNull())
)

silver_nba.write.mode("overwrite").saveAsTable(
    "main.default.silver_nba_stats"
)

display(silver_nba.limit(10))

# COMMAND ----------

#silver GYM Exercises 
gym = spark.table("main.default.bronze_gym_exercises")

display(gym.limit(5))
print(gym.columns)

# COMMAND ----------

from pyspark.sql.functions import coalesce, lit

silver_gym = (
    gym
    .dropDuplicates()
    .select(
        col("Title").alias("exercise_name"),
        col("Desc").alias("description"),
        col("Type").alias("exercise_type"),
        col("BodyPart").alias("body_part"),
        col("Equipment").alias("equipment"),
        col("Level").alias("difficulty_level")
    )
    .withColumn("exercise_name", trim(col("exercise_name")))
    .withColumn("description", coalesce(col("description"), lit("No description available")))
    .withColumn("difficulty_level", lower(trim(col("difficulty_level"))))
    .filter(col("exercise_name").isNotNull())
)

silver_gym.write.mode("overwrite").saveAsTable(
    "main.default.silver_gym_exercises"
)

display(silver_gym.limit(10))

# COMMAND ----------

#Silver Food Nutrients 
food_nutrients = spark.table("main.default.bronze_food_nutrients")

display(food_nutrients.limit(5))
print(food_nutrients.columns)

# COMMAND ----------

silver_food_nutrients = (
    food_nutrients
    .dropDuplicates()
)

silver_food_nutrients.write.mode("overwrite").saveAsTable(
    "main.default.silver_food_nutrients"
)

display(silver_food_nutrients.limit(10))

# COMMAND ----------

#Silver Food Portions 

food_portions = spark.table("main.default.bronze_food_portions")

silver_food_portions = (
    food_portions
    .dropDuplicates()
)

silver_food_portions.write.mode("overwrite").saveAsTable(
    "main.default.silver_food_portions"
)

display(silver_food_portions.limit(10))

# COMMAND ----------

#verifying the Silver tables -- 
spark.sql("SHOW TABLES IN main.default").show(truncate=False)

# COMMAND ----------

