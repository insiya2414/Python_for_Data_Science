-- select * from forbes_global_2010_2014;
SELECT 
    company,
    continent
FROM forbes_global_2010_2014
group by 1,2
order by MAX(profits) DESC
LIMIT 1