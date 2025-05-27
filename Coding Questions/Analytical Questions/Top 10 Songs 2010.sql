SELECT 
    year_rank,
    group_name,
    song_name
FROM billboard_top_100_year_end
WHERE year = '2010'
GROUP BY 1,2,3
ORDER BY 1 ASC
LIMIT 10