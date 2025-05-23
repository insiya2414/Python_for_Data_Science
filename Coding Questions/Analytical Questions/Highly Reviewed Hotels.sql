SELECT hotel_name,
       total_number_of_reviews
FROM hotel_reviews
GROUP BY 1, 2
ORDER BY 2 DESC