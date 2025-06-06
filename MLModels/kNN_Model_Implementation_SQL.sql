

SELECT COUNT(*) Total,UserID FROM  [ANALYTICS_DEV].[dbo].[VW_BOOK_RATING_SOURCE] 
--WHERE UserID='11676'
GROUP BY UserID
ORDER BY 1

--d(p,q)=SQRT(SQR(p1-q1)+SQR(p2-q2)+SQR(p3-q3)+..SQR(pn-qn) )
SELECT '224108','260067',SQRT(SUM(Book_Rating_Difference)) kNNScore 
FROM 
(
SELECT DISTINCT  p.UserID,p.ISBN,p.Book_Rating,q.UserID N_USERID,q.ISBN N_ISBN,ISNULL(q.Book_Rating,0) N_Book_Rating
,SQUARE(ISNULL(p.Book_Rating,0)-ISNULL(q.Book_Rating,0)) Book_Rating_Difference 
FROM [ANALYTICS_DEV].[dbo].[VW_BOOK_RATING_SOURCE] p LEFT JOIN 
(
SELECT DISTINCT ISBN,Book_Rating,UserID 
FROM  [ANALYTICS_DEV].[dbo].[VW_BOOK_RATING_SOURCE] 
WHERE USERID='260067'
)q ON p.ISBN=q.ISBN
WHERE p.UserID='224108'
)knn




SELECT '224108','260067',SQRT(SUM(Book_Rating_Difference)) kNNScore 
FROM
(
SELECT DISTINCT  x.UserID,x.ISBN,x.Book_Rating,z.UserID N_USERID,z.ISBN N_ISBN,ISNULL(z.Book_Rating,0) N_Book_Rating,
SQUARE(ISNULL(x.Book_Rating,0)-ISNULL(z.Book_Rating,0)) Book_Rating_Difference FROM 
[ANALYTICS_DEV].[dbo].[VW_BOOK_RATING_SOURCE] x LEFT JOIN 
(
SELECT DISTINCT ISBN,Book_Rating,UserID 
FROM  [ANALYTICS_DEV].[dbo].[VW_BOOK_RATING_SOURCE] 
WHERE USERID='260067'
)z ON x.ISBN=z.ISBN
WHERE x.UserID='224108'
)knn_score
GROUP BY UserID