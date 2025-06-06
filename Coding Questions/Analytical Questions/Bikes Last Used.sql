SELECT 
    bike_number,
    MAX(end_time) AS last_used_time
FROM 
    dc_bikeshare_q1_2012
GROUP BY 
    bike_number
ORDER BY 
    last_used_time DESC;