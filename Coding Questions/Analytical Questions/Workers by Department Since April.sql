SELECT department,
       COUNT(worker_id) AS num_of_workers
FROM worker
WHERE joining_date>= '2014-04-01'
GROUP BY department
ORDER BY COUNT(worker_id) DESC;