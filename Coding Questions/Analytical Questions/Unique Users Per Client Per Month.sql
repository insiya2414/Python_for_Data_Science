-- SQL
SELECT client_id, 
EXTRACT (MONTH FROM time_id) AS month,
COUNT(DISTINCT user_id) as users_num
FROM fact_events
GROUP BY client_id, EXTRACT (MONTH FROM time_id)

-- Python Pandas
import pandas as pd

result = (
    fact_events.groupby(
        [fact_events["client_id"], fact_events["time_id"].dt.month]
    )["user_id"]
    .nunique()
    .reset_index()
)
result = result.rename(columns={"time_id": "month", "user_id": "users_num"})
