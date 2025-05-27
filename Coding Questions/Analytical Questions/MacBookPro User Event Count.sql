SELECT 
    COUNT(user_id) AS number_of_users,
    event_name
FROM playbook_events
WHERE device IN ('MacBookPro', 'macbook pro')
GROUP BY 2
ORDER BY COUNT(event_type) DESC