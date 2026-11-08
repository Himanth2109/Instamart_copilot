-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;
-- Q2 three-table join
SELECT c.city,p.category,SUM(o.order_amount) AS revenue FROM orders o
JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY revenue DESC;
-- Q3 scalar subquery
SELECT * FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);
-- Q4 LEFT JOIN
SELECT c.customer_id FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;
-- Q5 two CTEs
WITH spend AS (SELECT customer_id,SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id),
average AS (SELECT AVG(total_spend) AS average_spend FROM spend)
SELECT spend.*,average_spend FROM spend CROSS JOIN average WHERE total_spend>average_spend;
-- Q6 city rankings
WITH spend AS (SELECT c.city,c.customer_id,SUM(o.order_amount) AS total_spend FROM customers c
JOIN orders o USING(customer_id) WHERE o.status='completed' GROUP BY c.city,c.customer_id)
SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS rank,
DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_number FROM spend;
-- Q7 cumulative daily revenue
WITH daily AS (SELECT SUBSTR(order_ts,1,10) AS day,SUM(order_amount) AS revenue FROM orders
WHERE status='completed' GROUP BY SUBSTR(order_ts,1,10))
SELECT *,SUM(revenue) OVER(ORDER BY day) AS cumulative_revenue FROM daily ORDER BY day;
