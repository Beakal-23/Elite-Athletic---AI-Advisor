# Databricks notebook source
# Adding_agent_tools
spark.sql("USE CATALOG main")
spark.sql("USE SCHEMA default")

# COMMAND ----------

#Validating Gold Tables
tables = [
    "gold_exercise_recommendations",
    "gold_basketball_benchmarks",
    "gold_nutrition_lookup"
]

for t in tables:
    print(t)
    spark.sql(f"SELECT * FROM {t} LIMIT 5").display()

# COMMAND ----------

# MAGIC %md
# MAGIC **Tool 1: Exercise recommendation**

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC
# MAGIC CREATE OR REPLACE FUNCTION main.default.recommend_exercises(
# MAGIC   target_goal STRING,
# MAGIC   target_body_part STRING
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   description STRING,
# MAGIC   body_part STRING,
# MAGIC   equipment STRING,
# MAGIC   goal_tag STRING,
# MAGIC   agent_text STRING
# MAGIC )
# MAGIC COMMENT 'Returns exercise recommendations based on athlete goals and body part focus.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC   description,
# MAGIC   body_part,
# MAGIC   equipment,
# MAGIC   goal_tag,
# MAGIC   agent_text
# MAGIC FROM main.default.gold_exercise_recommendations
# MAGIC WHERE lower(body_part) LIKE concat('%', lower(target_body_part), '%')
# MAGIC    OR lower(goal_tag) LIKE concat('%', lower(target_goal), '%')
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.recommend_exercises(
# MAGIC   'strength',
# MAGIC   'legs'
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.recommend_exercises('strength', 'legs');

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.recommend_exercises('explosive power', 'core');

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.gold_nutrition_lookup
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE main.default.gold_basketball_benchmarks;

# COMMAND ----------

# MAGIC %md
# MAGIC **. Tool 2: Nutrition lookup**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION main.default.lookup_nutrient(
# MAGIC   nutrient_query STRING
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   food_nutrient_id INT,
# MAGIC   nutrient_name STRING,
# MAGIC   adjusted_amount DOUBLE,
# MAGIC   lab_method_id INT
# MAGIC )
# MAGIC COMMENT 'Returns nutrient records matching a nutrient name query.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC   food_nutrient_id,
# MAGIC   nutrient_name,
# MAGIC   adjusted_amount,
# MAGIC   lab_method_id
# MAGIC FROM main.default.gold_nutrition_lookup
# MAGIC WHERE lower(nutrient_name) LIKE concat('%', lower(nutrient_query), '%')
# MAGIC LIMIT 20;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.lookup_nutrient('fat');

# COMMAND ----------

# DBTITLE 1,Tool 3: Nutrition Guidance
# MAGIC %md
# MAGIC **Tool 3: Nutrition Guidance Lookup**
# MAGIC
# MAGIC This tool provides goal-specific nutrition guidance for basketball athletes, including timing recommendations and educational text.

# COMMAND ----------

# DBTITLE 1,Create Nutrition Guidance Table
# MAGIC %sql
# MAGIC -- Create the gold_nutrition_guidance table
# MAGIC CREATE OR REPLACE TABLE main.default.gold_nutrition_guidance (
# MAGIC   guidance_id INT COMMENT 'Unique identifier for the guidance record',
# MAGIC   goal_tag STRING COMMENT 'Goal category: explosiveness, speed_agility, conditioning, strength, core, recovery',
# MAGIC   timing STRING COMMENT 'When to apply: pre-workout, post-workout, during-workout, daily, game-day',
# MAGIC   guidance_text STRING COMMENT 'Educational nutrition guidance text',
# MAGIC   age_group STRING COMMENT 'Target age group: youth, adult, all',
# MAGIC   injury_context STRING COMMENT 'Injury-specific guidance: none, recovery, general',
# MAGIC   priority INT COMMENT 'Display priority (1=highest)'
# MAGIC )
# MAGIC COMMENT 'Basketball-specific nutrition guidance organized by goals and timing';
# MAGIC
# MAGIC SELECT 'Table created successfully' AS status;

# COMMAND ----------

# DBTITLE 1,Populate Nutrition Guidance Data
# MAGIC %sql
# MAGIC -- Insert sample nutrition guidance records
# MAGIC INSERT INTO main.default.gold_nutrition_guidance VALUES
# MAGIC -- Explosiveness / Vertical Jump
# MAGIC (1, 'explosiveness', 'pre-workout', 'Eat a light meal with complex carbs (oatmeal, banana, whole grain toast) 2-3 hours before explosive training. This provides sustained energy for high-intensity plyometric work.', 'all', 'none', 1),
# MAGIC (2, 'explosiveness', 'post-workout', 'Within 30 minutes after training, consume protein (20-30g) with fast-acting carbs like fruit or sports drink to support muscle recovery and replenish glycogen for explosive movements.', 'all', 'none', 1),
# MAGIC (3, 'explosiveness', 'daily', 'Include lean proteins at each meal (chicken, fish, eggs, Greek yogurt) to support muscle development needed for explosive power. Aim for 0.6-0.8g protein per pound of body weight daily.', 'youth', 'none', 2),
# MAGIC
# MAGIC -- Speed & Agility
# MAGIC (4, 'speed_agility', 'pre-workout', 'Light carbs 60-90 minutes before speed work helps maintain energy without feeling heavy. Try crackers with peanut butter or a small smoothie.', 'all', 'none', 1),
# MAGIC (5, 'speed_agility', 'post-workout', 'Rehydrate immediately after speed/agility training. Water plus a piece of fruit or sports drink helps replace fluids and electrolytes lost during intense footwork drills.', 'all', 'none', 1),
# MAGIC (6, 'speed_agility', 'daily', 'Stay hydrated throughout the day (8-10 glasses of water). Proper hydration supports quick reactions, coordination, and prevents cramping during speed work.', 'all', 'none', 2),
# MAGIC
# MAGIC -- Conditioning & Endurance
# MAGIC (7, 'conditioning', 'pre-workout', 'For conditioning sessions lasting over 45 minutes, eat a balanced meal 2-3 hours before with carbs, moderate protein, and minimal fat. Think grilled chicken with rice and vegetables.', 'all', 'none', 1),
# MAGIC (8, 'conditioning', 'during-workout', 'For long conditioning workouts (60+ minutes), sip a sports drink or diluted juice to maintain blood sugar and hydration during extended cardio work.', 'all', 'none', 2),
# MAGIC (9, 'conditioning', 'post-workout', 'After endurance training, focus on carb replenishment within 2 hours. A turkey sandwich, pasta with lean protein, or rice bowl helps restore energy stores.', 'all', 'none', 1),
# MAGIC (10, 'conditioning', 'daily', 'Eat balanced meals throughout the day. Athletes doing regular conditioning need adequate carbs (whole grains, fruits, vegetables) to fuel sustained training.', 'all', 'none', 2),
# MAGIC
# MAGIC -- Strength Building
# MAGIC (11, 'strength', 'pre-workout', 'Eat a moderate meal 2-3 hours before strength training with both protein and carbs. Example: chicken breast with sweet potato, or eggs with whole grain toast.', 'all', 'none', 1),
# MAGIC (12, 'strength', 'post-workout', 'Protein is critical after strength work. Within 30-60 minutes, consume 20-30g protein (chicken, fish, protein shake, Greek yogurt) plus carbs to support muscle repair and growth.', 'all', 'none', 1),
# MAGIC (13, 'strength', 'daily', 'Spread protein intake across meals and snacks. For muscle building, aim for protein at breakfast, lunch, dinner, and possibly a snack. Include variety: meat, fish, eggs, dairy, beans, nuts.', 'youth', 'none', 2),
# MAGIC
# MAGIC -- Core Stability
# MAGIC (14, 'core', 'pre-workout', 'Light snack 60-90 minutes before core work. Avoid heavy meals that might cause discomfort during core exercises. Try an apple with almond butter or a small yogurt.', 'all', 'none', 1),
# MAGIC (15, 'core', 'daily', 'Core strength benefits from overall good nutrition. Include foods rich in magnesium (nuts, seeds, leafy greens) and calcium (dairy, fortified plant milk) to support muscle function.', 'all', 'none', 2),
# MAGIC
# MAGIC -- Recovery Nutrition
# MAGIC (16, 'recovery', 'post-workout', 'Recovery nutrition is essential. Chocolate milk is an excellent post-workout choice for youth athletes - provides protein, carbs, calcium, and hydration in one convenient drink.', 'youth', 'recovery', 1),
# MAGIC (17, 'recovery', 'daily', 'Prioritize sleep nutrition: eat a light dinner 2-3 hours before bed. Include foods with tryptophan (turkey, dairy, nuts) and complex carbs to support quality sleep needed for recovery.', 'all', 'recovery', 2),
# MAGIC (18, 'recovery', 'daily', 'For injury recovery, maintain protein intake to support tissue repair. Include anti-inflammatory foods like berries, fatty fish (salmon), leafy greens, and avoid excessive sugar or processed foods.', 'all', 'recovery', 1),
# MAGIC
# MAGIC -- Game Day Nutrition
# MAGIC (19, 'game-day', 'pre-workout', 'Eat a familiar, tested meal 3-4 hours before game time. Focus on carbs with moderate protein, low fat. Never try new foods on game day. Example: pasta with marinara and lean protein.', 'all', 'none', 1),
# MAGIC (20, 'game-day', 'during-workout', 'During games/tournaments, stay hydrated. Sip water or sports drink during breaks. For back-to-back games, quick snacks between games: banana, pretzels, granola bar.', 'all', 'none', 1),
# MAGIC (21, 'game-day', 'post-workout', 'After games, prioritize recovery nutrition within 30-60 minutes. Protein plus carbs helps muscles recover. Stay hydrated. Good choices: sandwich, wrap, or chocolate milk with fruit.', 'all', 'none', 1),
# MAGIC
# MAGIC -- General Youth Guidance
# MAGIC (22, 'general', 'daily', 'Eat breakfast every day. Youth athletes need morning fuel for school performance and afternoon practice. Include protein, whole grains, and fruit. Avoid skipping meals.', 'youth', 'none', 1),
# MAGIC (23, 'general', 'daily', 'Pack healthy snacks for school: trail mix, fruit, granola bars, cheese sticks, crackers with hummus. Having snacks available prevents poor food choices and maintains energy between meals.', 'youth', 'none', 2),
# MAGIC (24, 'general', 'daily', 'Limit sugary drinks and energy drinks. Water should be the primary beverage. Milk, 100% juice (in moderation), and sports drinks during intense activity are appropriate. Avoid soda and energy drinks.', 'youth', 'none', 1),
# MAGIC (25, 'general', 'daily', 'Focus on whole foods: fruits, vegetables, lean proteins, whole grains, low-fat dairy. Limit processed foods, fast food, and excessive sweets. Balance and variety support athletic performance.', 'all', 'none', 2);
# MAGIC
# MAGIC SELECT COUNT(*) AS records_inserted FROM main.default.gold_nutrition_guidance;

# COMMAND ----------

# DBTITLE 1,Create Nutrition Guidance UC Function
# MAGIC %sql
# MAGIC -- Create the Unity Catalog function for nutrition guidance lookup
# MAGIC CREATE OR REPLACE FUNCTION main.default.lookup_nutrition_guidance(
# MAGIC   goal_query STRING,
# MAGIC   injury_query STRING
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   guidance_id INT,
# MAGIC   goal_tag STRING,
# MAGIC   timing STRING,
# MAGIC   guidance_text STRING,
# MAGIC   age_group STRING,
# MAGIC   injury_context STRING,
# MAGIC   priority INT
# MAGIC )
# MAGIC COMMENT 'Returns nutrition guidance records matching goal and injury status, ordered by priority.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC   guidance_id,
# MAGIC   goal_tag,
# MAGIC   timing,
# MAGIC   guidance_text,
# MAGIC   age_group,
# MAGIC   injury_context,
# MAGIC   priority
# MAGIC FROM main.default.gold_nutrition_guidance
# MAGIC WHERE 
# MAGIC   (
# MAGIC     lower(goal_tag) LIKE concat('%', lower(goal_query), '%')
# MAGIC     OR goal_tag = 'general'
# MAGIC     OR goal_tag = 'game-day'
# MAGIC   )
# MAGIC   AND (
# MAGIC     injury_context = 'none'
# MAGIC     OR (lower(injury_query) != 'none' AND injury_context IN ('recovery', 'general'))
# MAGIC   )
# MAGIC ORDER BY priority ASC, guidance_id ASC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Test Nutrition Guidance Function - Explosiveness
# MAGIC %sql
# MAGIC -- Test the function: explosiveness goal with no injury
# MAGIC SELECT 
# MAGIC   goal_tag,
# MAGIC   timing,
# MAGIC   LEFT(guidance_text, 100) AS guidance_preview,
# MAGIC   age_group
# MAGIC FROM main.default.lookup_nutrition_guidance('explosiveness', 'none')
# MAGIC ORDER BY priority, timing;

# COMMAND ----------

# DBTITLE 1,Test Nutrition Guidance Function - Recovery
# MAGIC %sql
# MAGIC -- Test the function: conditioning goal with injury
# MAGIC SELECT 
# MAGIC   goal_tag,
# MAGIC   timing,
# MAGIC   LEFT(guidance_text, 100) AS guidance_preview,
# MAGIC   injury_context
# MAGIC FROM main.default.lookup_nutrition_guidance('conditioning', 'ankle sprain')
# MAGIC ORDER BY priority, timing;

# COMMAND ----------

# DBTITLE 1,Test Nutrition Guidance Function - General Youth
# MAGIC %sql
# MAGIC -- Test the function: general guidance for youth
# MAGIC SELECT 
# MAGIC   goal_tag,
# MAGIC   timing,
# MAGIC   LEFT(guidance_text, 100) AS guidance_preview,
# MAGIC   age_group
# MAGIC FROM main.default.lookup_nutrition_guidance('ball handling', 'none')
# MAGIC WHERE age_group IN ('youth', 'all')
# MAGIC ORDER BY priority, timing;

# COMMAND ----------

# MAGIC %md
# MAGIC **Basketball benchmark comparison**

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT position
# MAGIC FROM main.default.gold_basketball_benchmarks
# MAGIC ORDER BY position;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION main.default.get_basketball_benchmark(
# MAGIC   player_position STRING COMMENT 'Basketball position abbreviation such as PG, SG, SF, PF, C, PG-SG, or SG-PG.'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   position STRING,
# MAGIC   avg_points DOUBLE,
# MAGIC   avg_assists DOUBLE,
# MAGIC   avg_rebounds DOUBLE,
# MAGIC   avg_steals DOUBLE,
# MAGIC   avg_blocks DOUBLE,
# MAGIC   avg_minutes_played DOUBLE
# MAGIC )
# MAGIC COMMENT 'Returns basketball benchmark averages for a given basketball position from the gold_basketball_benchmarks table. Use this function whenever an athlete asks to compare performance against basketball benchmarks.'
# MAGIC RETURN
# MAGIC SELECT
# MAGIC   position,
# MAGIC   avg_points,
# MAGIC   avg_assists,
# MAGIC   avg_rebounds,
# MAGIC   avg_steals,
# MAGIC   avg_blocks,
# MAGIC   avg_minutes_played
# MAGIC FROM main.default.gold_basketball_benchmarks
# MAGIC WHERE lower(position) LIKE concat('%', lower(player_position), '%')
# MAGIC    OR lower(player_position) LIKE concat('%', lower(position), '%')
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM main.default.get_basketball_benchmark('G');

# COMMAND ----------

# MAGIC %md
# MAGIC **Finalizing The Agent Tool List**

# COMMAND ----------

UC_TOOL_NAMES = [
    "main.default.recommend_exercises",
    "main.default.lookup_nutrient",
    "main.default.lookup_nutrition_guidance",
    "main.default.get_basketball_benchmark"
]

# COMMAND ----------

# MAGIC %md
# MAGIC **Adding a Test Cell -- **

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main.default.recommend_exercises('strength', 'legs');
# MAGIC
# MAGIC SELECT * FROM main.default.lookup_nutrient('fat');
# MAGIC
# MAGIC SELECT * FROM main.default.get_basketball_benchmark('SG');

# COMMAND ----------

# MAGIC %md
# MAGIC **Tool Validation --**
# MAGIC
# MAGIC All three Unity Catalog tools were created and tested successfully.
# MAGIC
# MAGIC -  `recommend_exercises` retrieves exercise guidance from the gold exercise table.
# MAGIC - `lookup_nutrient` retrieves nutrient-level records from the nutrition table.
# MAGIC - `get_basketball_benchmark` retrieves basketball performance benchmarks by position.
# MAGIC
# MAGIC The basketball benchmark tool works best with position abbreviations such as `PG`, `SG`, `SF`, `PF`, and `C`.