import duckdb
import os
import pandas as pd

# ==========================================
# 1. SETUP AND CONFIGURATION
# ==========================================

# Define where the files live
RAW_DATA_PATH = os.path.join("data", "raw")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
OUTPUT_FILE_NAME = "training_data.csv"
OUTPUT_FILE_PATH = os.path.join(PROCESSED_DATA_PATH, OUTPUT_FILE_NAME)

# Make sure the output folder exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("--- Starting Feature Engineering Pipeline ---")

# Start DuckDB in memory (this makes processing very fast)
con = duckdb.connect(database=':memory:')

# ==========================================
# 2. LOAD RAW DATA
# ==========================================

try:
    print(f"Loading raw CSV files from {RAW_DATA_PATH}...")
    
    # Load each CSV file into a temporary SQL table
    con.execute(f"CREATE OR REPLACE TABLE profiles AS SELECT * FROM '{os.path.join(RAW_DATA_PATH, 'profile_data.csv')}'")
    con.execute(f"CREATE OR REPLACE TABLE activities AS SELECT * FROM '{os.path.join(RAW_DATA_PATH, 'activity_logs.csv')}'")
    con.execute(f"CREATE OR REPLACE TABLE trips AS SELECT * FROM '{os.path.join(RAW_DATA_PATH, 'trip_logs.csv')}'")
    con.execute(f"CREATE OR REPLACE TABLE incentives AS SELECT * FROM '{os.path.join(RAW_DATA_PATH, 'incentive_logs.csv')}'")
    con.execute(f"CREATE OR REPLACE TABLE interactions AS SELECT * FROM '{os.path.join(RAW_DATA_PATH, 'interaction_logs.csv')}'")
    
    print("Data loaded successfully.")

except Exception as e:
    print(f"Error loading data: {e}")
    # Cleanup connection before exiting
    con.close()
    exit()

# ==========================================
# 3. CREATE FEATURES (The Calculation Logic)
# ==========================================

print("\nExecuting feature calculations...")

# A) EFFICIENCY: earnings_per_hour_online (EPHO)
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_epho AS
WITH trip_stats AS (
    SELECT
        driver_id,
        SUM(fare + tip) AS total_trip_earnings
    FROM trips
    GROUP BY driver_id
),
activity_stats AS (
    SELECT
        driver_id,
        SUM(date_diff('second', session_start, session_end)) / 3600.0 AS total_hours_online
    FROM activities
    GROUP BY driver_id
)
SELECT
    a.driver_id,
    COALESCE(t.total_trip_earnings, 0) / NULLIF(a.total_hours_online, 0) AS avg_earnings_per_hour_online
FROM activity_stats a
LEFT JOIN trip_stats t ON a.driver_id = t.driver_id;
""")

# B) EFFICIENCY: trip_utilization_rate
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_utilization AS
WITH trip_time AS (
    SELECT
        driver_id,
        SUM(trip_duration_seconds) AS total_trip_seconds
    FROM trips
    GROUP BY driver_id
),
online_time AS (
    SELECT
        driver_id,
        SUM(date_diff('second', session_start, session_end)) AS total_online_seconds
    FROM activities
    GROUP BY driver_id
)
SELECT
    o.driver_id,
    COALESCE(t.total_trip_seconds, 0)::DOUBLE / NULLIF(o.total_online_seconds, 0) AS trip_utilization_rate
FROM online_time o
LEFT JOIN trip_time t ON o.driver_id = t.driver_id;
""")

# C) STRATEGY: surge_reliance_score
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_surge AS
SELECT
    driver_id,
    SUM(fare - (fare / NULLIF(surge_multiplier, 0))) / NULLIF(SUM(fare), 0) AS surge_reliance_score
FROM trips
GROUP BY driver_id;
""")

# D) STRATEGY: premium_trip_ratio and is_premium_capable
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_premium_strategy AS
WITH capability AS (
    SELECT
        driver_id,
        CASE
            WHEN vehicle_dispatchability ILIKE '%UberBlack%'
              OR vehicle_dispatchability ILIKE '%Premier%'
            THEN 1
            ELSE 0
        END AS is_premium_capable
    FROM profiles
),
execution AS (
    SELECT
        driver_id,
        SUM(CASE WHEN trip_type IN ('UberBlack', 'Premier') THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(COUNT(trip_id), 0) AS premium_trip_ratio
    FROM trips
    GROUP BY driver_id
)
SELECT
    c.driver_id,
    c.is_premium_capable,
    COALESCE(e.premium_trip_ratio, 0.0) AS premium_trip_ratio
FROM capability c
LEFT JOIN execution e ON c.driver_id = e.driver_id;
""")

# E) COMMITMENT: peak_hour_driver_score
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_peak_hours AS
WITH trips_with_local_time AS (
    SELECT
        driver_id,
        trip_id,
        request_time AT TIME ZONE 'America/New_York' AS local_request_time
    FROM trips
)
SELECT
    t.driver_id,
    SUM(
        CASE
            WHEN dayofweek(t.local_request_time) IN (1, 2, 3, 4) AND (
                (strftime(t.local_request_time, '%H:%M:%S') >= '07:00:00' AND strftime(t.local_request_time, '%H:%M:%S') < '10:00:00') OR
                (strftime(t.local_request_time, '%H:%M:%S') >= '16:00:00' AND strftime(t.local_request_time, '%H:%M:%S') < '19:00:00')
            ) THEN 1
            WHEN dayofweek(t.local_request_time) = 5 AND (
                (strftime(t.local_request_time, '%H:%M:%S') >= '07:00:00' AND strftime(t.local_request_time, '%H:%M:%S') < '09:00:00') OR
                (strftime(t.local_request_time, '%H:%M:%S') >= '16:00:00')
            ) THEN 1
            WHEN dayofweek(t.local_request_time) IN (6, 0) AND (
                (strftime(t.local_request_time, '%H:%M:%S') >= '09:00:00') OR
                (strftime(t.local_request_time, '%H:%M:%S') >= '00:00:00' AND strftime(t.local_request_time, '%H:%M:%S') < '05:00:00')
            ) THEN 1
            ELSE 0
        END
    )::DOUBLE / NULLIF(COUNT(t.trip_id), 0) AS peak_hour_driver_score
FROM trips_with_local_time t
GROUP BY 1;
""")

# F) COMMITMENT: session_regularity (Standard Deviation of gaps)
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_regularity AS
WITH session_gaps AS (
    SELECT
        driver_id,
        DATE_DIFF('hour',
                  LAG(session_end) OVER (PARTITION BY driver_id ORDER BY session_start),
                  session_start) AS time_between_sessions_hours
    FROM activities
)
SELECT
    driver_id,
    STDDEV(time_between_sessions_hours) AS session_regularity
FROM session_gaps
GROUP BY driver_id;
""")

# G) INCENTIVES: quest_completion_rate
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_quest_completion AS
SELECT
    driver_id,
    SUM(CASE WHEN status = 'completed' AND incentive_type = 'Quest' THEN 1 ELSE 0 END)::DOUBLE
    / NULLIF(SUM(CASE WHEN incentive_type = 'Quest' THEN 1 ELSE 0 END), 0) AS quest_completion_rate
FROM incentives
GROUP BY driver_id;
""")

# H) INCENTIVES: incentive_reliance_pct
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_incentive_reliance AS
WITH bonus_earnings AS (
    SELECT
        driver_id,
        SUM(bonus_amount) AS total_bonus
    FROM incentives
    GROUP BY driver_id
),
trip_earnings AS (
    SELECT
        driver_id,
        SUM(fare) AS total_fare,
        SUM(tip) AS total_tip
    FROM trips
    GROUP BY driver_id
)
SELECT
    COALESCE(b.driver_id, t.driver_id) AS driver_id,
    COALESCE(b.total_bonus, 0.0) AS total_bonus_earned,
    COALESCE(t.total_fare, 0.0) AS total_fare_earned,
    COALESCE(t.total_tip, 0.0) AS total_tip_earned,
    COALESCE(b.total_bonus, 0.0) /
    NULLIF(
        COALESCE(b.total_bonus, 0.0) +
        COALESCE(t.total_fare, 0.0) +
        COALESCE(t.total_tip, 0.0)
    , 0.0) AS incentive_reliance_pct
FROM bonus_earnings b
FULL JOIN trip_earnings t ON b.driver_id = t.driver_id;
""")

# I) LOYALTY: pro_tier_status
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_loyalty AS
SELECT
    driver_id,
    CASE
        WHEN current_tier='Diamond' THEN 0
        WHEN current_tier='Platinum' THEN 1 
        WHEN current_tier='Gold' THEN 2
        ELSE 3
    END AS pro_tier_status
FROM profiles;
""")

# J) FRUSTRATION: cancellation_rate
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_cancellation AS
SELECT
    driver_id,
    SUM(CASE WHEN event_type = 'trip_cancelled' THEN 1 ELSE 0 END)::DOUBLE
    / NULLIF(SUM(CASE WHEN event_type = 'trip_accepted' THEN 1 ELSE 0 END), 0) AS cancellation_rate
FROM interactions
GROUP BY driver_id;
""")

# K) FRUSTRATION: acceptance_rate
con.execute("""
CREATE OR REPLACE TEMP TABLE feat_acceptance AS
SELECT
    driver_id,
    SUM(CASE WHEN event_type = 'trip_accepted' THEN 1 ELSE 0 END)::DOUBLE
    / NULLIF(SUM(CASE WHEN event_type IN ('trip_accepted', 'trip_ignored') THEN 1 ELSE 0 END), 0) AS acceptance_rate
FROM interactions
GROUP BY driver_id;
""")

print("All temporary feature tables created.")

# ==========================================
# 4. FINAL JOIN AND SAVE
# ==========================================
 
# 3. Create the Final Master Table
print("\nJoining all features into final table...")
final_query = """
CREATE OR REPLACE TABLE profile_final_features AS
SELECT
    -- 1. Base Driver Profile and Target Variables
    p.driver_id,
    p.signup_date,
    p.avg_rating,
    p.current_tier,
    p.Churned, -- The target variable for Model 2
    
    -- 2. Efficiency Features
    epho.avg_earnings_per_hour_online,
    util.trip_utilization_rate,
    
    -- 3. Strategy Features
    surge.surge_reliance_score,
    prem.is_premium_capable,    
    prem.premium_trip_ratio,    
    
    -- 4. Commitment Features
    peak.peak_hour_driver_score,
    reg.session_regularity,
    
    -- 5. Incentive Features
    quest.quest_completion_rate,
    rely.incentive_reliance_pct,
    
    -- 6. Loyalty & Frustration Features
    loy.pro_tier_status,
    cancel.cancellation_rate,
    accept.acceptance_rate

FROM profiles p
LEFT JOIN feat_epho epho ON p.driver_id = epho.driver_id
LEFT JOIN feat_utilization util ON p.driver_id = util.driver_id
LEFT JOIN feat_surge surge ON p.driver_id = surge.driver_id
LEFT JOIN feat_premium_strategy prem ON p.driver_id = prem.driver_id
LEFT JOIN feat_peak_hours peak ON p.driver_id = peak.driver_id
LEFT JOIN feat_regularity reg ON p.driver_id = reg.driver_id
LEFT JOIN feat_quest_completion quest ON p.driver_id = quest.driver_id
LEFT JOIN feat_incentive_reliance rely ON p.driver_id = rely.driver_id
LEFT JOIN feat_loyalty loy ON p.driver_id = loy.driver_id
LEFT JOIN feat_cancellation cancel ON p.driver_id = cancel.driver_id
LEFT JOIN feat_acceptance accept ON p.driver_id = accept.driver_id;
"""
con.execute(final_query)

# 4. Save the Result to a CSV File
final_df = con.sql("SELECT * FROM profile_final_features").df()
# *** FIX APPLIED HERE: Using OUTPUT_FILE_PATH instead of PROCESSED_DATA_PATH ***
final_df.to_csv(OUTPUT_FILE_PATH, index=False)

print(f"\nSuccessfully created and saved final feature table with {len(final_df)} rows.")
# *** FIX APPLIED HERE: Using OUTPUT_FILE_PATH for printing the path ***
print(f"Output file: {OUTPUT_FILE_PATH}")

# 5. Cleanup
con.close()
print("--- Feature Engineering Pipeline Complete ---")