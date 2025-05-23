SELECT 
    AVG(bathrooms) AS avg_bath, 
    AVG(bedrooms) AS avg_bed,
    city,
    property_type
FROM airbnb_search_details
GROUP BY 3,4
ORDER BY 1,2
