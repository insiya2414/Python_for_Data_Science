SELECT hotel_name,
       reviewer_score,
       COUNT(reviewer_score) AS score
FROM hotel_reviews
WHERE hotel_name = 'Hotel Arena'
GROUP BY 1,2
Order by 2 ASC